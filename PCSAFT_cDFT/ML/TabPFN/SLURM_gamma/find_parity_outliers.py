"""
find_parity_outliers.py
-----------------------
Identify and report outliers in the TabPFN gamma parity plot.

Outliers are defined as points whose absolute residual exceeds a threshold
set by IQR-based fencing (Tukey method) on the chosen split's residuals.
When --split all is used, the parity plot colour-codes points by split
(train/val/test) with outliers shown as filled markers.

Usage:
    python find_parity_outliers.py [--k <IQR multiplier>] [--split <train|test|val|all>]
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 – activates the style

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PRED_CSV     = os.path.join(SCRIPT_DIR, "TabPFN_gamma_predictions.csv")
DATASET_CSV  = os.path.join(SCRIPT_DIR, "..", "..", "interfacial_results_dataset_A1.csv")
OUTPUT_DIR   = SCRIPT_DIR

Z_COLS = [
    "z_carbon dioxide", "z_hydrogen", "z_nitrogen", "z_argon",
    "z_methane", "z_oxygen", "z_water", "z_carbon monoxide",
    "z_nitrogen dioxide", "z_nitrogen oxide", "z_sulfur dioxide",
    "z_hydrogen sulfide", "z_propane", "z_ethane",
]

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--k",     type=float, default=3.0,
                    help="IQR multiplier for fence (default: 3.0)")
parser.add_argument("--split", type=str,   default="test",
                    choices=["train", "test", "val", "all"],
                    help="Which split to analyse (default: test)")
args = parser.parse_args()

# ── Load data ─────────────────────────────────────────────────────────────────
preds   = pd.read_csv(PRED_CSV)
df_orig = pd.read_csv(DATASET_CSV)

if args.split != "all":
    subset = preds[preds["split"] == args.split].copy()
else:
    subset = preds.copy()

subset["residual"]     = subset["actual"] - subset["predicted"]
subset["abs_residual"] = subset["residual"].abs()

# ── IQR-based outlier fence ───────────────────────────────────────────────────
Q1, Q3 = subset["abs_residual"].quantile([0.25, 0.75])
IQR     = Q3 - Q1
fence   = Q3 + args.k * IQR

outliers    = subset[subset["abs_residual"] > fence].copy()
non_outlier = subset[subset["abs_residual"] <= fence]

print(f"\n{'='*60}")
print(f"Outlier detection  |  split={args.split}  |  k={args.k}")
print(f"{'='*60}")
print(f"  Total points analysed : {len(subset)}")
print(f"  IQR fence threshold   : {fence:.6f}  (Q3={Q3:.6f}, IQR={IQR:.6f})")
print(f"  Outliers found        : {len(outliers)}\n")

# ── Merge with original dataset for full context ──────────────────────────────
if not outliers.empty:
    orig_rows = df_orig.iloc[outliers["idx"].values].copy()
    orig_rows.index = outliers.index          # align for concat
    detail = pd.concat([outliers[["idx", "actual", "predicted",
                                   "residual", "abs_residual"]], orig_rows], axis=1)

    # Composition: keep only non-zero z_ columns per row
    z_present = [c for c in Z_COLS if c in df_orig.columns and (orig_rows[c] != 0).any()]

    print("Outlier details:")
    report_cols = ["idx", "actual", "predicted", "residual",
                   "temperature", "pressure"] + z_present + ["source_id"]
    print(detail[report_cols].to_string(index=False))

    # Also show P_bubble proximity for context
    if "P_bubble" in detail.columns:
        detail["P_over_Pbubble"] = detail["pressure"] / detail["P_bubble"]
        print("\n  P / P_bubble for outliers:")
        print(detail[["idx", "temperature", "pressure",
                       "P_bubble", "P_over_Pbubble"]].to_string(index=False))

    # Save outlier table
    out_csv = os.path.join(OUTPUT_DIR, "parity_outliers.csv")
    detail.to_csv(out_csv, index=False)
    print(f"\nOutlier table saved → {out_csv}")
else:
    print("No outliers found at this threshold.")

# ── Parity plot with outliers annotated ──────────────────────────────────────
SPLIT_STYLE = {
    "train": dict(color="#4878CF", marker="o", label="Train"),   # blue
    "val":   dict(color="#6ACC65", marker="s", label="Val"),      # green
    "test":  dict(color="#D65F5F", marker="^", label="Test"),     # red
}
OUTLIER_EDGE = {
    "train": "#1a3a6e",
    "val":   "#276024",
    "test":  "#7a1a1a",
}

plt.style.use(["science", "nature"])
fig, ax = plt.subplots(figsize=(5, 5))

lo = subset[["actual", "predicted"]].min().min() * 0.97
hi = subset[["actual", "predicted"]].max().max() * 1.03
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, zorder=0)

splits_present = subset["split"].unique()

for sp in ["train", "val", "test"]:
    if sp not in splits_present:
        continue
    style = SPLIT_STYLE[sp]
    sp_normal   = non_outlier[non_outlier["split"] == sp]
    sp_outliers = outliers[outliers["split"] == sp] if not outliers.empty else pd.DataFrame()

    # Normal points
    if not sp_normal.empty:
        ax.scatter(sp_normal["actual"], sp_normal["predicted"],
                   s=6, alpha=0.35,
                   color=style["color"], marker=style["marker"],
                   label=f"{style['label']} ({len(sp_normal)})",
                   rasterized=True)

    # Outlier points — same colour, filled + dark edge
    if not sp_outliers.empty:
        ax.scatter(sp_outliers["actual"], sp_outliers["predicted"],
                   s=50, alpha=1.0, zorder=5,
                   color=style["color"], marker=style["marker"],
                   edgecolors=OUTLIER_EDGE[sp], linewidths=0.8,
                   label=f"{style['label']} outlier ({len(sp_outliers)})")

        for _, row in sp_outliers.iterrows():
            orig = df_orig.iloc[int(row["idx"])]
            src  = int(orig["source_id"]) if "source_id" in orig else ""
            ax.annotate(f"idx={int(row['idx'])}\nsrc={src}",
                        xy=(row["actual"], row["predicted"]),
                        xytext=(6, -14), textcoords="offset points",
                        fontsize=5, color=OUTLIER_EDGE[sp],
                        arrowprops=dict(arrowstyle="-", color=OUTLIER_EDGE[sp], lw=0.5))

ax.set_xlabel(r"Actual $\gamma$ (mN m$^{-1}$)")
ax.set_ylabel(r"Predicted $\gamma$ (mN m$^{-1}$)")
ax.set_title(f"Parity plot – {args.split} split(s)  (outlier fence: k={args.k})")
ax.legend(fontsize=6, loc="upper left", framealpha=0.7)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")

plot_path = os.path.join(OUTPUT_DIR, "parity_outliers_annotated.png")
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
print(f"Annotated parity plot saved → {plot_path}")
plt.close()
