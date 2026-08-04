#!/usr/bin/env python
"""
export_svgp_predictions.py
==========================
Re-run inference with the stored SVGP residual model and cache its predictions.

The SVGP run saved only ``SVGP_residual_model.pt`` (weights + scaler + quantile
transformer) and a metrics JSON -- unlike the GPR run, which also wrote
``GPR_predictions.npz`` and ``GPR_gamma_reconstructed.npz``.  Rebuilding the
Figure 9 panels therefore needed the predictions back.  Everything upstream is
deterministic (fixed seed for the split, fitted transformers in the
checkpoint), so this reproduces them exactly and writes the two missing npz
files with the same names and keys the GPR run used.

The reconstruction follows the notebook (SVGP_GPR_RESIDUAL.ipynb, cells 12 and
25): predictions come back in the normalised, quantile-transformed space and
are mapped to mN/m in three steps, with the predictive standard deviation
carried through the quantile transform by its numerical Jacobian.

Usage (inside the py_A6 venv):
    python export_svgp_predictions.py
"""

from __future__ import annotations

import json
import os

import gpytorch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(HERE, "SVGP_RESIDUAL_OUTPUTS")
DATA_CSV = os.path.join(HERE, "..", "CombinedDatasetSEC_A4.csv")
SEED = 4555525             # as in the notebook; fixes the 70/15/15 split
QT_EPS = 1e-4              # step of the numerical Jacobian of the QT inverse


class SVGPModel(gpytorch.models.ApproximateGP):
    """Stochastic variational GP with an ARD Matern-5/2 kernel (as trained)."""

    def __init__(self, inducing_points: torch.Tensor):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5,
                                          ard_num_dims=inducing_points.shape[1])
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def predict_batched(model, likelihood, X: torch.Tensor, batch_size: int = 2048):
    """Batched posterior mean and standard deviation, in normalised units."""
    model.eval()
    likelihood.eval()
    means, stds = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, len(X), batch_size):
            pred = likelihood(model(X[i:i + batch_size]))
            means.append(pred.mean.numpy())
            stds.append(pred.stddev.numpy())
    return np.concatenate(means), np.concatenate(stds)


def load_dataframe() -> pd.DataFrame:
    """The training frame, rebuilt exactly as the notebook built it."""
    df = pd.read_csv(DATA_CSV)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df = df.dropna(subset=["gamma_wsd", "gamma_cDFT_minus_wsd_uncorrected",
                           "gamma_wsd_UC"]).copy()
    df["gamma_cDFT_UC"] = df["gamma_wsd_UC"] + df["gamma_cDFT_minus_wsd_uncorrected"]
    df["gamma_cDFT"] = df["gamma_wsd"] + df["gamma_cDFT_minus_wsd_corrected"]
    df["gamma_base"] = df["gamma_wsd_UC"]
    df["residual"] = df["gamma_cDFT_UC"] - df["gamma_wsd_UC"]
    return df


def main() -> int:
    ckpt = torch.load(os.path.join(OUTPUT_FOLDER, "SVGP_residual_model.pt"),
                      weights_only=False)
    features, scaler, qt = ckpt["features"], ckpt["scaler"], ckpt["qt"]
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]

    df = load_dataframe()
    X, y = df[features], df["residual"]

    # 70/15/15, same seed as training -- this is what makes the split identical.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED)
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED)

    inducing = ckpt["model_state_dict"]["variational_strategy.inducing_points"]
    model = SVGPModel(torch.zeros_like(inducing))
    model.load_state_dict(ckpt["model_state_dict"])
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.load_state_dict(ckpt["likelihood_state_dict"])

    def qt_inverse(a):
        return qt.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()

    def predict(Xs):
        Xt = torch.from_numpy(scaler.transform(Xs).astype(np.float32))
        mean_n, std_n = predict_batched(model, likelihood, Xt)
        mean_qt = mean_n * y_std + y_mean
        # d y_orig / d y_qt, evaluated by central difference
        jac = np.abs(qt_inverse(mean_qt + QT_EPS)
                     - qt_inverse(mean_qt - QT_EPS)) / (2 * QT_EPS)
        return qt_inverse(mean_qt), std_n * y_std * jac

    splits = {"train": (X_train, y_train), "test": (X_test, y_test),
              "val": (X_val, y_val)}
    pred, std, recon = {}, {}, {}
    for name, (Xs, ys) in splits.items():
        pred[name], std[name] = predict(Xs)
        recon[name] = (df.loc[Xs.index, "gamma_cDFT"].to_numpy(),
                       df.loc[Xs.index, "gamma_base"].to_numpy() + pred[name])

    np.savez(os.path.join(OUTPUT_FOLDER, "SVGP_predictions.npz"),
             **{f"y_{k}_pred": pred[k] for k in splits},
             **{f"y_{k}_std": std[k] for k in splits})
    np.savez(os.path.join(OUTPUT_FOLDER, "SVGP_gamma_reconstructed.npz"),
             **{f"gamma_cDFT_{k}": recon[k][0] for k in splits},
             **{f"gamma_pred_{k}": recon[k][1] for k in splits})

    # Compare against the metrics the training run recorded: if the split or
    # the transforms had drifted, these would not line up.
    with open(os.path.join(OUTPUT_FOLDER, "SVGP_residual_metrics.json")) as fh:
        stored = json.load(fh)
    print(f"{'split':<6}{'n':>7}{'RMSE':>12}{'stored':>12}"
          f"{'R2(gamma)':>12}{'stored':>12}")
    for name, (Xs, ys) in splits.items():
        rmse = float(np.sqrt(np.mean((ys.to_numpy() - pred[name]) ** 2)))
        r2g = r2_score(*recon[name])
        print(f"{name:<6}{len(ys):>7}{rmse:>12.6f}"
              f"{stored[f'{name}_rmse']:>12.6f}"
              f"{r2g:>12.7f}{stored[f'gamma_cDFT_r2_{name}']:>12.7f}")
        mae = mean_absolute_error(ys, pred[name])
        assert abs(mae - stored[f"{name}_mae"]) < 1e-6, f"{name} MAE drifted"

    print(f"wrote SVGP_predictions.npz and SVGP_gamma_reconstructed.npz "
          f"in {OUTPUT_FOLDER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
