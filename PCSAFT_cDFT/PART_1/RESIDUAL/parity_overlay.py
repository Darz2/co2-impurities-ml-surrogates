# %% [markdown]
# Cross-model parity overlay on the reconstructed gamma_cDFT scale (mN/m).
# Loads per-split prediction CSVs for each model and renders a 3-panel parity plot.
#
# Expected CSV columns (each row = one sample):
#     gamma_cDFT_actual, gamma_cDFT_pred
#
# Models whose prediction CSVs aren't present are skipped (with a printed note).
# To add a model: write three CSVs (train/val/test) in the convention below
# with those two column names, and register it in MODEL_CSV_PATHS.

# %% imports
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% paths
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

# Per-model {split: csv_path}
MODEL_CSV_PATHS = {
    "SR": {
        "train": HERE / "SR/Mixtures/SR_MIXTURES_OUTPUTS/SR_train_predictions_eps_base.csv",
        "val":   HERE / "SR/Mixtures/SR_MIXTURES_OUTPUTS/SR_validation_predictions_eps_base.csv",
        "test":  HERE / "SR/Mixtures/SR_MIXTURES_OUTPUTS/SR_test_predictions_eps_base.csv",
    },
    "GPR": {
        "train": HERE / "GPR/GPR_RESIDUAL_OUTPUTS/GPR_train_predictions.csv",
        "val":   HERE / "GPR/GPR_RESIDUAL_OUTPUTS/GPR_validation_predictions.csv",
        "test":  HERE / "GPR/GPR_RESIDUAL_OUTPUTS/GPR_test_predictions.csv",
    },
    "SVGP": {
        "train": HERE / "SVGP/SVGP_RESIDUAL_OUTPUTS/SVGP_train_predictions.csv",
        "val":   HERE / "SVGP/SVGP_RESIDUAL_OUTPUTS/SVGP_validation_predictions.csv",
        "test":  HERE / "SVGP/SVGP_RESIDUAL_OUTPUTS/SVGP_test_predictions.csv",
    },
}

OUT_DIR = HERE / "MODEL_COMPARISON"
OUT_DIR.mkdir(exist_ok=True)

# %% load thermoift PLOT_SETTINGS
PLOT_SETTINGS_PATH = (HERE / "../../thermoift/src/thermoift/PLOT_SETTINGS.py").resolve()
ps = None
if PLOT_SETTINGS_PATH.exists():
    _spec = importlib.util.spec_from_file_location("thermoift_plot_settings", PLOT_SETTINGS_PATH)
    ps = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(ps)
    plt.rcParams["font.family"]      = ps.graphic_font
    plt.rcParams["mathtext.fontset"] = ps.math_font
    plt.rcParams["text.usetex"]      = True

# %% metric helpers
def rmse(a, p):
    return float(np.sqrt(np.mean((a - p) ** 2)))

def r2(a, p):
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

# %% load each model's predictions (gamma_cDFT scale)
def load_model(name, paths):
    out = {}
    for split, p in paths.items():
        if not p.exists():
            return None, f"{name}: missing {p.name}"
        df = pd.read_csv(p)
        if "gamma_cDFT_actual" not in df.columns or "gamma_cDFT_pred" not in df.columns:
            return None, f"{name}/{split}: needs columns gamma_cDFT_actual, gamma_cDFT_pred"
        out[split] = (df["gamma_cDFT_actual"].to_numpy(), df["gamma_cDFT_pred"].to_numpy())
    return out, None

loaded = {}
for name, paths in MODEL_CSV_PATHS.items():
    data, err = load_model(name, paths)
    if data is None:
        print(f"SKIP {err}")
        continue
    loaded[name] = data

if not loaded:
    raise SystemExit("No model prediction CSVs found; nothing to plot.")

# %% global axis range across all models / splits
all_vals = np.concatenate([
    arr for d in loaded.values() for split in d.values() for arr in split
])
lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
pad = 0.02 * (hi - lo)
lo -= pad; hi += pad

# %% plot — one panel per model
SPLIT_STYLE = {
    "train": {"label": "Train",      "marker": "o", "color": "#1f77b4"},
    "val":   {"label": "Validation", "marker": "D", "color": "#2ca02c"},
    "test":  {"label": "Test",       "marker": "^", "color": "#d62728"},
}

n_models = len(loaded)
fig, axes = plt.subplots(1, n_models, figsize=(4.8 * n_models, 4.6), squeeze=False)
axes = axes[0]

xlabel = r"Actual $\Gamma_{\mathrm{cDFT}}$ / [mN m$^{-1}$]" if ps else "Actual gamma_cDFT [mN/m]"
ylabel = r"Predicted $\Gamma_{\mathrm{cDFT}}$ / [mN m$^{-1}$]" if ps else "Predicted gamma_cDFT [mN/m]"

for ax, (name, data) in zip(axes, loaded.items()):
    for split, (a, p) in data.items():
        style = SPLIT_STYLE[split]
        ax.scatter(a, p, s=14, alpha=0.55, marker=style["marker"],
                   edgecolor="black", linewidth=0.2,
                   color=style["color"], label=style["label"])
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # annotation: RMSE / R² per split (test split is what matters most)
    rows = [name]
    for split in ("train", "val", "test"):
        a, p = data[split]
        rows.append(f"{SPLIT_STYLE[split]['label']}: RMSE={rmse(a, p):.3g}  R²={r2(a, p):.3f}")
    txt = "\n".join(rows)
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.6", lw=0.4))
    ax.set_title(name)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)

fig.suptitle("Parity on reconstructed gamma_cDFT", y=1.02)
fig.tight_layout()

png_path = OUT_DIR / "parity_overlay_gamma_cDFT.png"
pdf_path = OUT_DIR / "parity_overlay_gamma_cDFT.pdf"
fig.savefig(png_path, dpi=200, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
