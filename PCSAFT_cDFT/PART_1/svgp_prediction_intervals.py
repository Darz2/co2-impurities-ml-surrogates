#!/usr/bin/env python
"""
svgp_prediction_intervals.py
============================
Predictive intervals of the hybrid WSD + SVGP residual model for the IFT.

Reviewer request (R2): "It would be helpful to include one example showing
95 % prediction intervals from TabPFN to illustrate model confidence."

For P_bubble that request is answered directly from TabPFN
(``tabpfn_prediction_intervals.py``), because the archived P_bubble model is a
plain ``TabPFNRegressor`` whose forward pass emits a density over the target --
the mean is one readout of it, the quantiles are another.

The archived gamma model is not.  It is a ``DecisionTreeTabPFNRegressor`` (the
``dt_pfn`` architecture of tabpfn_extensions: a CART tree with a separate TabPFN
fitted at each node, the node outputs combined in logit space with adaptive
pruning).  Its ``predict`` returns point values only -- there is no quantile
path, because what comes back is no longer one TabPFN density but a pruned
aggregate of several.  Reconstructing a mixture quantile from it would be our
own construction, not the model's.

The IFT does, however, already have a probabilistic model in this work: the
hybrid WSD + SVGP residual correction of Section 3.6, whose predictive standard
deviation is the quantity plotted in Fig. 10d.  This script reads intervals off
that model instead, which is a genuine posterior predictive rather than a
reconstruction.

How the interval is formed
--------------------------
The SVGP is trained on the residual eps = gamma_cDFT - gamma_WSD in a
normalised, quantile-transformed space.  ``likelihood(model(x))`` is a Normal in
that space, including the observation-noise term, so a central interval is

    [ mu - z * sigma ,  mu + z * sigma ]      z = Phi^-1((1 + c) / 2)

and its endpoints are mapped back to mN/m through the *inverse quantile
transform*.  That transform is monotone, so mapping the endpoints maps the
quantiles exactly -- unlike propagating sigma through its Jacobian, which
linearises and forces the interval to be symmetric.  The resulting intervals in
mN/m are asymmetric, as the TabPFN ones are.

The baseline gamma_WSD is deterministic, so the interval on the reconstructed
IFT is the interval on the residual, shifted:

    gamma_hat  = gamma_WSD + eps_hat
    interval   = gamma_WSD + [eps_lo, eps_hi]

Coverage is therefore identical in residual space and in IFT space; it is
reported against gamma_cDFT_UC = gamma_wsd_UC + residual, the pairing the model
was actually trained on.

Outputs
-------
  FIGURE11/SVGP_gamma_intervals.csv    one row per evaluated state point:
                                       split, feed, T, P, reference IFT, the
                                       WSD baseline, predictive mean and the
                                       interval ladder
  FIGURE11/SVGP_gamma_coverage.json    empirical coverage of the central
                                       intervals on the test set, interval
                                       widths, and the reproduction check

Usage
-----
    python svgp_prediction_intervals.py
    python svgp_prediction_intervals.py --out /tmp/scratch
"""

from __future__ import annotations

import argparse
import json
import os

import gpytorch
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Paths and run definition                                                     #
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
RESIDUAL = os.path.join(HERE, "RESIDUAL")
DATASET = os.path.join(RESIDUAL, "CombinedDatasetSEC_A4.csv")
RUN_DIR = os.path.join(RESIDUAL, "SVGP", "SVGP_RESIDUAL_OUTPUTS")
CHECKPOINT = os.path.join(RUN_DIR, "SVGP_residual_model.pt")
METRICS = os.path.join(RUN_DIR, "SVGP_residual_metrics.json")
OUTPUT_DIR = os.path.join(HERE, "FIGURE11")

TARGET = "gamma"
UNIT = "mN/m"
SEED = 4555525          # SVGP_GPR_RESIDUAL.ipynb; fixes the 70/15/15 split

CENTRAL_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

# Feeds of Table 1 of the manuscript, matched to the dataset by composition --
# the same convention, tolerance and column order as the TabPFN script, except
# that this dataset spells the species with underscores.
TABLE1_COLUMNS = [
    "z_carbon_dioxide", "z_hydrogen", "z_argon", "z_nitrogen",
    "z_methane", "z_oxygen", "z_carbon_monoxide", "z_hydrogen_sulfide",
]
TABLE1_FEEDS = {
    1: [0.97, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0],
    2: [0.95, 0.0313, 0.0, 0.0, 0.0039, 0.0, 0.0, 0.0148],
    3: [0.98, 0.0058, 0.0091, 0.0011, 0.0029, 0.0011, 0.0, 0.0],
    4: [0.96, 0.0200, 0.0, 0.0080, 0.0040, 0.0040, 0.0020, 0.0020],
    5: [0.97, 0.0075, 0.0040, 0.0079, 0.0099, 3.79e-5, 6.99e-4, 3.90e-7],
}
FEED_MATCH_TOL = 1e-4   # feed 4 does not match at this tolerance


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #

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


def load_model(ckpt: dict) -> tuple:
    inducing = ckpt["model_state_dict"]["variational_strategy.inducing_points"]
    model = SVGPModel(torch.zeros_like(inducing))
    model.load_state_dict(ckpt["model_state_dict"])
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.load_state_dict(ckpt["likelihood_state_dict"])
    model.eval()
    likelihood.eval()
    return model, likelihood


def posterior(model, likelihood, X: torch.Tensor, batch: int = 2048) -> tuple:
    """Predictive mean and standard deviation, in the normalised target space."""
    means, stds = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, len(X), batch):
            pred = likelihood(model(X[i:i + batch]))
            means.append(pred.mean.numpy())
            stds.append(pred.stddev.numpy())
    return np.concatenate(means), np.concatenate(stds)


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #

def load_dataframe() -> pd.DataFrame:
    """The training frame, rebuilt exactly as the notebook built it."""
    df = pd.read_csv(DATASET)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df = df.dropna(subset=["gamma_wsd", "gamma_cDFT_minus_wsd_uncorrected",
                           "gamma_wsd_UC"]).copy()
    df["gamma_cDFT_UC"] = df["gamma_wsd_UC"] + df["gamma_cDFT_minus_wsd_uncorrected"]
    df["gamma_cDFT"] = df["gamma_wsd"] + df["gamma_cDFT_minus_wsd_corrected"]
    df["gamma_base"] = df["gamma_wsd_UC"]
    df["residual"] = df["gamma_cDFT_UC"] - df["gamma_wsd_UC"]
    return df


def load_split(df: pd.DataFrame, features: list) -> tuple:
    """Rebuild the exact 70/15/15 split of the archived run."""
    X, y = df[features], df["residual"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED)
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED)
    return X_train, y_train, X_test, y_test, X_val, y_val


def match_table1_feeds(df: pd.DataFrame) -> dict[int, int]:
    """Table 1 feed number -> ``source_id``, for the feeds that are in the set."""
    comp = df.groupby("source_id")[TABLE1_COLUMNS].first()
    found = {}
    for feed, z in TABLE1_FEEDS.items():
        dz = np.abs(comp.to_numpy() - np.asarray(z)).max(axis=1)
        i = int(dz.argmin())
        if dz[i] <= FEED_MATCH_TOL:
            found[feed] = int(comp.index[i])
        else:
            print(f"  Table 1 feed {feed}: no match in the dataset "
                  f"(closest source_id {comp.index[i]}, max|dz| = {dz[i]:.1e})")
    return found


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #

def coverage_table(frame: pd.DataFrame) -> list[dict]:
    """Empirical coverage of each central interval, on the given rows."""
    rows = []
    y = frame["actual"].to_numpy()
    for lvl in CENTRAL_LEVELS:
        lo = frame[f"lo{lvl:.2f}"].to_numpy()
        hi = frame[f"hi{lvl:.2f}"].to_numpy()
        inside = (y >= lo) & (y <= hi)
        width = hi - lo
        rows.append({
            "nominal": lvl,
            "empirical": float(inside.mean()),
            "n": int(len(y)),
            "mean_width": float(width.mean()),
            "median_width": float(np.median(width)),
        })
    return rows


def reproduction_check(frame: pd.DataFrame) -> dict | None:
    """RMSE against the metrics the training run recorded.

    Only the test split is compared: it is evaluated in full here, whereas the
    validation rows carried along are just those of the example feeds, so their
    RMSE is not comparable with the stored full-split number.
    """
    if not os.path.exists(METRICS):
        return None
    with open(METRICS) as fh:
        stored = json.load(fh)
    test = frame[frame.split == "test"]
    return {
        "split": "test",
        "n": int(len(test)),
        "rmse": float(np.sqrt(np.mean((test["actual"] - test["mean"]) ** 2))),
        "rmse_stored": stored.get("test_rmse"),
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUTPUT_DIR,
                        help="output directory (default: FIGURE11/)")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(CHECKPOINT, weights_only=False)
    features, scaler, qt = ckpt["features"], ckpt["scaler"], ckpt["qt"]
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
    print(f"Target {TARGET} ({UNIT}), SVGP residual model, seed {SEED}")
    print(f"  features: {features}")

    df = load_dataframe()
    X_train, y_train, X_test, y_test, X_val, y_val = load_split(df, features)
    print(f"  train {len(X_train)}  test {len(X_test)}  val {len(X_val)}")

    feeds = match_table1_feeds(df)
    print(f"  Table 1 feeds present: "
          f"{', '.join(f'{k} -> source_id {v}' for k, v in feeds.items())}")

    # Held-out rows only: an interval around a training point illustrates
    # nothing.  The example-feed rows are kept alongside the test split so the
    # example panel and the coverage panel come from one pass.
    on_feed = df["source_id"].isin(feeds.values())
    feed_idx = (X_test.index[on_feed.loc[X_test.index]]
                .append(X_val.index[on_feed.loc[X_val.index]]))
    idx = feed_idx.append(X_test.index.difference(feed_idx))
    split = pd.Series("test", index=idx)
    split.loc[idx.intersection(X_val.index)] = "val"

    print(f"Predicting {len(idx)} points "
          f"({(split == 'test').sum()} test, {(split == 'val').sum()} val; "
          f"{len(feed_idx)} of them on the example feeds) ...")

    model, likelihood = load_model(ckpt)
    Xt = torch.from_numpy(
        scaler.transform(df.loc[idx, features]).astype(np.float32))
    mean_n, std_n = posterior(model, likelihood, Xt)

    # Normalised -> quantile-transformed space.  The inverse quantile transform
    # is monotone, so mapping interval endpoints through it maps the quantiles
    # exactly; the intervals come back asymmetric in mN/m.
    mean_qt = mean_n * y_std + y_mean
    std_qt = std_n * y_std
    inv = lambda a: qt.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()

    rows = df.loc[idx]
    base = rows["gamma_base"].to_numpy()
    out = pd.DataFrame(
        {
            "split": split.loc[idx],
            "source_id": rows["source_id"],
            "temperature": rows["T"],
            "pressure": rows["P"],
            "actual": rows["gamma_cDFT_UC"],
            "baseline": base,
            "mean": base + inv(mean_qt),
        },
        index=idx,
    )
    for lvl in CENTRAL_LEVELS:
        z = norm.ppf((1.0 + lvl) / 2.0)
        out[f"lo{lvl:.2f}"] = base + inv(mean_qt - z * std_qt)
        out[f"hi{lvl:.2f}"] = base + inv(mean_qt + z * std_qt)
    out.index.name = "idx"

    csv_path = os.path.join(args.out, f"SVGP_{TARGET}_intervals.csv")
    out.to_csv(csv_path)
    print(f"Wrote {csv_path}")

    test = out[out.split == "test"]
    summary = {
        "model": "WSD + SVGP residual correction",
        "run": os.path.relpath(RUN_DIR, HERE),
        "target": TARGET,
        "unit": UNIT,
        "seed": SEED,
        "device": "cpu",
        "n_train": int(len(X_train)),
        "n_inducing": int(ckpt["n_inducing"]),
        "features": features,
        "levels": CENTRAL_LEVELS,
        "coverage_test": coverage_table(test),
        "table1_feed_to_source_id": feeds,
        "test_rmse": float(np.sqrt(np.mean(
            (test["mean"] - test["actual"]) ** 2))),
        "reproduction_check": reproduction_check(out),
    }
    json_path = os.path.join(args.out, f"SVGP_{TARGET}_coverage.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {json_path}")

    print("\nCoverage of the central intervals (test split):")
    for row in summary["coverage_test"]:
        print(f"  nominal {row['nominal']:.0%}  empirical "
              f"{row['empirical']:.1%}  median width "
              f"{row['median_width']:.4g} {UNIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
