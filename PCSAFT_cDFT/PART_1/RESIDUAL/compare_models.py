# %% [markdown]
# Cross-model RMSE / R² comparison
# Loads GPR_residual_metrics.json, SVGP_residual_metrics.json and
# SR_residual_metrics.json and plots grouped bar charts.
# Metrics shown are on the reconstructed gamma_cDFT scale (mN/m, R²),
# so SR (eps_base target) is directly comparable to GPR/SVGP (residual target).

# %% imports
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# %% paths
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

METRICS_PATHS = {
    "GPR":  HERE / "GPR/GPR_RESIDUAL_OUTPUTS/GPR_residual_metrics.json",
    "SVGP": HERE / "SVGP/SVGP_RESIDUAL_OUTPUTS/SVGP_residual_metrics.json",
    "SR":   HERE / "SR/Mixtures/SR_MIXTURES_OUTPUTS/SR_residual_metrics.json",
}

OUT_DIR = HERE / "MODEL_COMPARISON"
OUT_DIR.mkdir(exist_ok=True)

# %% load thermoift PLOT_SETTINGS (same convention as plot_residual_3d.py)
PLOT_SETTINGS_PATH = (HERE / "../../thermoift/src/thermoift/PLOT_SETTINGS.py").resolve()
ps = None
if PLOT_SETTINGS_PATH.exists():
    _spec = importlib.util.spec_from_file_location("thermoift_plot_settings", PLOT_SETTINGS_PATH)
    ps = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(ps)
    plt.rcParams["font.family"]      = ps.graphic_font
    plt.rcParams["mathtext.fontset"] = ps.math_font
    plt.rcParams["text.usetex"]      = True

# %% load metrics
metrics = {}
for name, path in METRICS_PATHS.items():
    if not path.exists():
        print(f"WARNING: metrics file missing for {name}: {path}")
        continue
    with open(path) as f:
        metrics[name] = json.load(f)

if not metrics:
    raise SystemExit("No metrics files found; nothing to plot.")

# %% pick the keys we need (reconstructed gamma_cDFT scale)
SPLITS = ["train", "val", "test"]
RMSE_KEYS = {s: f"gamma_cDFT_rmse_{s}" for s in SPLITS}
R2_KEYS   = {s: f"gamma_cDFT_r2_{s}"   for s in SPLITS}

def get_metric(m, key):
    return m.get(key, np.nan)

models = list(metrics.keys())
rmse_vals = np.array([[get_metric(metrics[m], RMSE_KEYS[s]) for s in SPLITS] for m in models])
r2_vals   = np.array([[get_metric(metrics[m], R2_KEYS[s])   for s in SPLITS] for m in models])

# %% grouped bar chart helper
def grouped_bars(ax, vals, models, splits, ylabel, title, *, log=False):
    n_models = len(models)
    n_splits = len(splits)
    x = np.arange(n_models)
    width = 0.8 / n_splits
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_splits))

    for j, s in enumerate(splits):
        offsets = x + (j - (n_splits - 1) / 2) * width
        bars = ax.bar(offsets, vals[:, j], width=width, label=s.capitalize(),
                      color=colors[j], edgecolor="black", linewidth=0.4)
        for rect, v in zip(bars, vals[:, j]):
            if np.isfinite(v):
                ax.annotate(f"{v:.3g}",
                            xy=(rect.get_x() + rect.get_width() / 2, v),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="y", alpha=0.3)
    if log:
        ax.set_yscale("log")

# %% plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

grouped_bars(
    axes[0], rmse_vals, models, SPLITS,
    ylabel=r"RMSE of $\Gamma_{\mathrm{cDFT}}$ / [mN m$^{-1}$]" if ps else "RMSE of gamma_cDFT [mN/m]",
    title="RMSE (reconstructed gamma)",
    log=True,
)

grouped_bars(
    axes[1], r2_vals, models, SPLITS,
    ylabel=r"$R^2$ of $\Gamma_{\mathrm{cDFT}}$" if ps else "R² of gamma_cDFT",
    title="R² (reconstructed gamma)",
)
axes[1].set_ylim(0.0, 1.05)

fig.suptitle("Residual ML models — cross-model comparison", y=1.02)
fig.tight_layout()

png_path = OUT_DIR / "model_comparison_rmse_r2.png"
pdf_path = OUT_DIR / "model_comparison_rmse_r2.pdf"
fig.savefig(png_path, dpi=200, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

# %% also dump the aggregated table to CSV for downstream use
import csv
summary_csv = OUT_DIR / "model_comparison_summary.csv"
with open(summary_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "split", "gamma_cDFT_rmse_mNm", "gamma_cDFT_r2", "gamma_cDFT_mae_mNm"])
    for m in models:
        for s in SPLITS:
            w.writerow([
                m, s,
                get_metric(metrics[m], f"gamma_cDFT_rmse_{s}"),
                get_metric(metrics[m], f"gamma_cDFT_r2_{s}"),
                get_metric(metrics[m], f"gamma_cDFT_mae_{s}_mNm"),
            ])
print(f"Saved: {summary_csv}")
