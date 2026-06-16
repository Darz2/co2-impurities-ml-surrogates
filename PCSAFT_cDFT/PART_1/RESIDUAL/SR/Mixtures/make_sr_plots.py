#!/usr/bin/env python3
# ============================================================
#  Post-processing: parity + residual plots for the SR mixtures runs.
#  Regenerates, for BOTH feature sets, the same figures the report uses:
#     SR_parity_{target}.(png|pdf)      — predicted vs actual
#     SR_residual_{target}.(png|pdf)    — residual vs predicted (±RMSE bands)
#
#  Standalone port of SR_R2_RMSE_plots.ipynb, looping over V1 and V2.
#  Run AFTER Mixture.jl and Mixture_V2.jl have written their prediction CSVs.
#
#  Usage (inside the py_A6 venv):
#     python make_sr_plots.py
# ============================================================
from pathlib import Path
import numpy as np
import pandas as pd
from thermoift import MLPostprocessing, PLOT_SETTINGS as ps

HERE = Path(__file__).resolve().parent

# ---- figure / tick styling (override the thermoift PLOT_SETTINGS globals) ----
ps.plot_size      = (4, 3)   # was (4, 3): larger figure
ps.tick_labelsize = 14         # was 10:    larger tick labels
ps.tick_length    = 5          # was 4:     longer major ticks
ps.tick_width     = 1.0        # was 0.75:  thicker major ticks

# ---- legend styling passed to the plot calls below ----
LEGEND_FONTSIZE = 10
LEGEND_BOLD     = True

# ---- per-axis tick styling (x and y can be set independently) ----
# Forwarded to ax.tick_params(axis="x"/"y", ...); overrides the ps.* tick
# globals above for that axis only. Edit each dict separately.
XTICK_PARAMS = {"labelsize": 14, "length": 5, "width": 1.0}   # x-axis ticks
YTICK_PARAMS = {"labelsize": 14, "length": 5, "width": 1.0}   # y-axis ticks

# (output folder, label) for each feature set
RUNS = [
    ("SR_MIXTURES_OUTPUTS",    "V1"),
    ("SR_MIXTURES_OUTPUTS_V2", "V2"),
]
TARGETS = ["gamma", "delta_gamma", "eps_base"]
SOURCE_CSV = HERE / "../../CombinedDatasetSEC_A4.csv"


def rmse(a, p):
    return float(np.sqrt(np.mean((a - p) ** 2)))


def r2(a, p):
    a, p = np.asarray(a), np.asarray(p)
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def build_gamma_base_lookup(source_path):
    src = pd.read_csv(source_path)
    gb = src["gamma_wsd_UC"].to_numpy(dtype=np.float64)
    gc = (src["gamma_wsd_UC"] + src["gamma_cDFT_minus_wsd_uncorrected"]).to_numpy(dtype=np.float64)
    eps = (gc - gb) / gb
    ok = np.isfinite(eps) & np.isfinite(gb) & (gb != 0.0)
    eps, gb = eps[ok], gb[ok]
    lookup = {}
    for k, g in zip(eps.tolist(), gb.tolist()):
        lookup.setdefault(k, g)
    return lookup


def recover_gamma_base(eps_values, lookup):
    return np.array([lookup.get(float(v), np.nan) for v in eps_values], dtype=np.float64)


def load_all_targets(csv_paths):
    out = {"gamma": {}, "delta_gamma": {}, "eps_base": {}}
    need_lookup = any(
        not (set(pd.read_csv(p, nrows=0).columns) & {"gamma_base", "gamma_cDFT_actual"})
        for p in csv_paths.values()
    )
    lookup = build_gamma_base_lookup(SOURCE_CSV) if (need_lookup and SOURCE_CSV.exists()) else None

    for split, path in csv_paths.items():
        df = pd.read_csv(path)
        eps_a = df["Actual"].to_numpy(dtype=np.float64)
        eps_p = df["Predicted"].to_numpy(dtype=np.float64)
        out["eps_base"][split] = (eps_a, eps_p)

        gb = None
        if "gamma_base" in df.columns:
            gb = df["gamma_base"].to_numpy(dtype=np.float64)
        elif "gamma_cDFT_actual" in df.columns:
            gb = df["gamma_cDFT_actual"].to_numpy(dtype=np.float64) / (1.0 + eps_a)
        elif lookup is not None:
            gb = recover_gamma_base(eps_a, lookup)
        if gb is None:
            continue

        keep = np.isfinite(gb)
        if keep.sum() == 0:
            continue
        gb_k, eps_a_k, eps_p_k = gb[keep], eps_a[keep], eps_p[keep]
        if {"gamma_cDFT_actual", "gamma_cDFT_pred"}.issubset(df.columns):
            g_a = df["gamma_cDFT_actual"].to_numpy(dtype=np.float64)[keep]
            g_p = df["gamma_cDFT_pred"].to_numpy(dtype=np.float64)[keep]
        else:
            g_a = gb_k * (1.0 + eps_a_k)
            g_p = gb_k * (1.0 + eps_p_k)
        out["gamma"][split] = (g_a, g_p)
        out["delta_gamma"][split] = (gb_k * eps_a_k, gb_k * eps_p_k)

    needed = set(csv_paths.keys())
    return {t: ps for t, ps in out.items() if set(ps.keys()) == needed}


def make_plots(target_key, target_data, out_dir):
    thermoift_target = {"gamma": "gamma", "delta_gamma": "delta_gamma", "eps_base": "delta_gamma"}[target_key]
    post = MLPostprocessing(
        y_true=target_data["test"][0],
        y_pred=target_data["test"][1],
        target=thermoift_target,
        datasets={k: target_data[k] for k in ("train", "test", "val")},
    )
    if target_key == "eps_base":
        post._get_target_label = lambda: r"$\varepsilon_{\gamma}$ / $[-]$"
        post._get_target_symbol = lambda: r"\varepsilon_{\gamma}"
    post.plot_parity(
        save_path=f"SR_parity_{target_key}", folder=str(out_dir),
        legend_fontsize=LEGEND_FONTSIZE, legend_bold=LEGEND_BOLD,
        xtick_params=XTICK_PARAMS, ytick_params=YTICK_PARAMS,
    )
    post.plot_residual_vs_predicted(
        save_path=f"SR_residual_{target_key}", folder=str(out_dir), y_range=(-0.5, 0.5),
        legend_fontsize=LEGEND_FONTSIZE, legend_bold=LEGEND_BOLD,
        xtick_params=XTICK_PARAMS, ytick_params=YTICK_PARAMS,
    )


def main():
    for folder, label in RUNS:
        out_dir = HERE / folder
        csv_paths = {
            "train": out_dir / "SR_train_predictions_eps_base.csv",
            "val":   out_dir / "SR_validation_predictions_eps_base.csv",
            "test":  out_dir / "SR_test_predictions_eps_base.csv",
        }
        if not all(p.exists() for p in csv_paths.values()):
            print(f"[{label}] SKIP — prediction CSVs not found in {folder}")
            continue
        print(f"\n=== {label} : {folder} ===")
        data = load_all_targets(csv_paths)
        for tgt in (t for t in TARGETS if t in data):
            make_plots(tgt, data[tgt], out_dir)
            for s in ("train", "val", "test"):
                a, p = data[tgt][s]
                print(f"  {tgt:12s} {s:5s}  RMSE={rmse(a, p):.4g}  R²={r2(a, p):.4f}  n={len(a)}")
        print(f"  saved parity + residual plots in {folder}")


if __name__ == "__main__":
    main()
