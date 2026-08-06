#!/usr/bin/env python
"""
build_figure9.py
================
Unified builder for every panel of Figure 9 (hybrid IFT-residual models).

Figure 9 is a 3x2 grid: three residual models (rows) x two diagnostics
(columns), all for the reconstructed interfacial tension
``gamma_hat = gamma_wsd + Delta_gamma_pred``.  Like Figure 4, the panels used
to come from three unrelated places -- a Julia/notebook pipeline for SR and one
notebook each for GPR and SVGP -- so this script rebuilds all six from the
stored predictions with one style definition.

The style is Figure 4's, imported from ``build_figure4`` rather than copied, so
the two figures cannot drift apart:

* Colour is not the only channel carrying the split.  Train / Test / Validation
  differ in marker shape (square / circle / triangle) as well as hue, and the
  +/-RMSE bands differ in dash pattern (dashed vs dash-dot), so the panels stay
  readable under dichromacy and in greyscale.
* Grey / blue / red is not a red-green confusion pair; it survives protanopia,
  deuteranopia and tritanopia.  Measured (CIEDE2000 on the drawn marker faces),
  the worst of the three pairs scores 12 under protanopia against 5.5 for the
  published palette, whose Train (lightgray) and Test (lightblue) were the real
  failure: 0.6 apart in greyscale, i.e. indistinguishable in a mono print.  The
  marker edges now carry that pair 13 L* units apart, and the shapes carry it
  regardless of colour.
* Legend, axis-label and tick-label sizes are Figure 4's: 12 pt legends, 18 pt
  axis labels, 13 pt ticks.  These panels sit at 0.4\\textwidth, i.e. reduced
  by only ~1.3x against Figure 4's ~2.4x, so the same point sizes land
  *larger* on the page here -- ~9 pt legend text against ~5 pt before.
* The +/-RMSE legend is placed by the same scan as Figure 4 (see
  ``build_figure4.place_rmse_legend``): it goes in whichever band of the panel
  hides the fewest markers.  The Test/Validation marker key is kept inside the
  residual panels, unlike Figure 4 where it was moved to the caption for space;
  at 0.4\\textwidth there is room for it, and it is fed to the placement scan
  as an occupied region so the +/-RMSE legend routes around it.
* The parity legend gets the same band scan (``place_parity_legend``), and its
  split names are shortened to the "Val." the +/-RMSE legend already uses.  The
  free space in a parity panel is the triangle above the diagonal, so a legend
  fits only while its width plus its height stays under the panel; with the
  full names at 12 pt that sum is 1.06, and no placement avoids the data.  The
  short names bring it to 0.93.
* Panel (e) is relabelled from "gamma" to "gamma_hat" to agree with the other
  five, which already used the hat.

Everything else -- data, splits, RMSE and R^2 values, axis ranges, panel order
-- is unchanged with respect to the published version of the figure.

The SVGP run stored only its trained model, so its predictions are recovered by
``RESIDUAL/SVGP/export_svgp_predictions.py``, which must be run once before
panels (e) and (f) can be built.

Usage
-----
    python build_figure9.py                 # -> FIGURE9/{with,no}_legend/Figure9a..f
    python build_figure9.py --overleaf      # also copy with_legend into OverleafDir
    python build_figure9.py --panels a,b    # rebuild a subset (quick iteration)
    python build_figure9.py --dpi 300       # draft resolution
    python build_figure9.py --legend off    # only the legend-free set
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys

import matplotlib
matplotlib.use("pgf")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

import thermoift.PLOT_SETTINGS as ps

# Style, fonts and the legend-placement/saving machinery are Figure 4's.
from build_figure4 import (
    apply_font_settings,
    BAND_LINEWIDTH,
    LABELPAD,
    LABEL_FONTSIZE,
    LEGEND_SUBDIR,
    MARKER_ALPHA,
    MARKER_LINEWIDTH,
    MARKER_SIZE,
    RMSE_FONTSIZE,
    RMSE_LEGEND_WITH_UNIT,
    SPLIT_STYLE,
    TICK_LABELSIZE,
    bold,
    legend_box,
    place_rmse_legend,
    save_variants,
    scan_band,
    unit_inner,
    visible_marker_frac,
)

# OpenType Latin Modern through lualatex, no Type 3 fonts.  Importing
# build_figure4 has already applied these; the call is repeated so the setting
# is visible here too and does not depend on that side effect surviving a
# refactor.  See build_figure4 for why.
apply_font_settings()

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "FIGURE9")
RESIDUAL_DIR = os.path.join(HERE, "RESIDUAL")
OVERLEAF_DIR = os.path.abspath(
    os.path.join(HERE, "..", "..", "OverleafDir", "A6-Draft-Overleaf")
)

# --------------------------------------------------------------------------- #
# Split style                                                                  #
# --------------------------------------------------------------------------- #

# Test and Validation are Figure 4's, unchanged.  Train is only ever shown in
# the parity panels, so it needs no band dash pattern -- a third shape and a
# neutral grey are enough, and grey keeps it visually subordinate to the two
# splits the reader is meant to compare.
TRAIN_STYLE = dict(
    label=r"$\mathrm{Train}$",
    short="Train",
    marker="s",                     # square
    face="#BBBBBB",                 # light grey
    edge="#4D4D4D",                 # dark grey
)

PARITY_SPLITS = {
    "train": TRAIN_STYLE,
    "test": SPLIT_STYLE["test"],
    "val": SPLIT_STYLE["val"],
}

N_SIGMA = 2.0               # error bars on the parity panels, as published
ERRORBAR_LINEWIDTH = 0.6
ERRORBAR_CAPSIZE = 1.5
PARITY_MARGIN = 0.5         # data-unit padding on the parity axis limits

SHOW_SPLIT_LEGEND = True    # Test/Validation marker key inside the residual
                            # panels; there is room for it at 0.4\textwidth
SPLIT_LEGEND_GRID = 7       # points per side used to mask the split legend
                            # from the +/-RMSE placement scan

# The reconstructed target.  All six panels share it, so unlike Figure 4 there
# is a single label rather than one per row.
TARGET_LABEL = r"$\hat{\gamma}$ / $[\mathrm{mN\,m^{-1}}]$"
TARGET_SYMBOL = r"\hat{\gamma}"
PARITY_TICK = 5             # major tick spacing, both parity axes
RESIDUAL_X_TICK = 5
RESIDUAL_Y_RANGE = (-3, 3)  # common across the three models, as published
RESIDUAL_Y_TICK = 1


# --------------------------------------------------------------------------- #
# Data sources                                                                 #
# --------------------------------------------------------------------------- #

def load_sr() -> dict[str, tuple]:
    """SR predictions from the Julia run's per-split CSVs.

    ``gamma_cDFT_actual``/``gamma_cDFT_pred`` are the reconstructed values the
    Julia pipeline wrote out; SR carries no predictive variance, hence no std.
    """
    folder = os.path.join(RESIDUAL_DIR, "SR", "Mixtures", "SR_MIXTURES_OUTPUTS")
    names = {"train": "SR_train_predictions_eps_base.csv",
             "test": "SR_test_predictions_eps_base.csv",
             "val": "SR_validation_predictions_eps_base.csv"}
    out = {}
    for split, name in names.items():
        df = pd.read_csv(os.path.join(folder, name))
        out[split] = (df["gamma_cDFT_actual"].to_numpy(),
                      df["gamma_cDFT_pred"].to_numpy(), None)
    return out


def load_npz(folder: str, prefix: str) -> dict[str, tuple]:
    """GPR/SVGP predictions from the npz pair their runs wrote out.

    The std is the posterior standard deviation of the *residual*, which
    carries over to the reconstructed value unchanged: gamma_hat is the
    residual plus a deterministic baseline.
    """
    recon = np.load(os.path.join(folder, f"{prefix}_gamma_reconstructed.npz"))
    preds = np.load(os.path.join(folder, f"{prefix}_predictions.npz"))
    return {
        split: (recon[f"gamma_cDFT_{split}"], recon[f"gamma_pred_{split}"],
                preds[f"y_{split}_std"])
        for split in ("train", "test", "val")
    }


MODELS = {
    "SR": dict(
        name="SR",
        load=load_sr,
        source="RESIDUAL/SR/Mixtures/SR_MIXTURES_OUTPUTS",
    ),
    "GPR": dict(
        name="GPR",
        load=lambda: load_npz(
            os.path.join(RESIDUAL_DIR, "GPR", "SLURM_GPR_residual_5000"), "GPR"),
        source="RESIDUAL/GPR/SLURM_GPR_residual_5000",
    ),
    "SVGP": dict(
        name="SVGP",
        load=lambda: load_npz(
            os.path.join(RESIDUAL_DIR, "SVGP", "SVGP_RESIDUAL_OUTPUTS"), "SVGP"),
        source="RESIDUAL/SVGP/SVGP_RESIDUAL_OUTPUTS",
    ),
}

# panel letter -> (model key, panel kind).  Row order follows the manuscript.
PANELS = {
    "a": ("SR", "parity"),
    "b": ("SR", "residual"),
    "c": ("GPR", "parity"),
    "d": ("GPR", "residual"),
    "e": ("SVGP", "parity"),
    "f": ("SVGP", "residual"),
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, as reported in the parity legends."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def marker_size_pt() -> float:
    """Figure 4's scatter area (pt^2) as an ``errorbar`` markersize (pt)."""
    return math.sqrt(MARKER_SIZE)


def legend_mask_points(fig, ax, legend) -> np.ndarray:
    """Data-coordinate grid covering ``legend``, for the placement scan.

    ``place_rmse_legend`` scores candidate positions by how many plotted points
    they cover, and knows nothing about other artists.  Filling the split
    legend's box with synthetic points makes the scan treat it as occupied, so
    the +/-RMSE legend routes around it instead of landing on top.
    """
    fig.canvas.draw()
    bb = legend.get_window_extent(fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    gx, gy = np.meshgrid(np.linspace(x0, x1, SPLIT_LEGEND_GRID),
                         np.linspace(y0, y1, SPLIT_LEGEND_GRID))
    return np.column_stack([gx.ravel(), gy.ravel()])


def place_parity_legend(fig, ax, handles, points: np.ndarray,
                        legend_bg: dict) -> tuple[str, int]:
    """Put the R^2 legend in whichever band of the panel hides fewest points.

    The parity cloud lies along the diagonal, so the free space is the upper
    left and the lower right corner; which of the two is roomier depends on
    where the cloud ends and on how far the error bars reach.  This picks
    between them exactly the way ``build_figure4.place_rmse_legend`` picks a
    band for the +/-RMSE legend -- the labels are wide enough at 12 pt to cross
    the parity line if simply pinned to a corner.
    """
    def make(anchor=None):
        kw = (dict(loc="lower left", bbox_to_anchor=anchor,
                   bbox_transform=ax.transAxes) if anchor else dict(loc="best"))
        leg = ax.legend(
            handles=handles,
            prop={"size": RMSE_FONTSIZE, "weight": "bold"},
            ncol=1, handlelength=1.0, handletextpad=0.4,
            **kw, **legend_bg,
        )
        leg.set_zorder(6)
        return leg

    probe = make()
    raw, infl = legend_box(fig, ax, probe)
    probe.remove()

    frac = visible_marker_frac(ax, points)
    over, hits, x0, y0 = min(scan_band(frac, *infl, vert)
                             for vert in ("lower", "upper"))
    anchor = (x0 + (infl[0] - raw[0]) / 2.0, y0 + (infl[1] - raw[1]) / 2.0)
    make(anchor)

    placement = (f"@({anchor[0]:.2f},{anchor[1]:.2f})"
                 + (f" [overflows {over:.2f}]" if over > 0 else ""))
    return placement, hits


def style_axes(ax) -> None:
    """Tick locators, sizes and minor ticks shared by both panel kinds."""
    ps.apply_axis_style(ax)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)
    ax.tick_params(axis="both", which="minor", length=3)


# --------------------------------------------------------------------------- #
# Panels                                                                       #
# --------------------------------------------------------------------------- #

def draw_parity(ax, splits: dict[str, tuple]) -> tuple[list, list, str]:
    """Actual vs predicted, one series per split, with +/-2 sigma error bars."""
    handles, scores, plotted = [], [], []
    all_values = []

    for key, style in PARITY_SPLITS.items():
        y_true, y_pred, y_std = splits[key]
        all_values.append(np.concatenate([y_true, y_pred]))
        plotted.append(np.column_stack([y_true, y_pred]))
        score = r2_score(y_true, y_pred)
        scores.append(f"{style['short']}={score:.4f}")

        if y_std is None:
            ax.scatter(
                y_true, y_pred,
                marker=style["marker"], s=MARKER_SIZE, alpha=MARKER_ALPHA,
                facecolors=style["face"], edgecolors=style["edge"],
                linewidths=MARKER_LINEWIDTH, zorder=1,
            )
        else:
            ax.errorbar(
                y_true, y_pred, yerr=N_SIGMA * y_std,
                fmt=style["marker"], markersize=marker_size_pt(),
                alpha=MARKER_ALPHA, color=style["edge"],
                markerfacecolor=style["face"], markeredgecolor=style["edge"],
                markeredgewidth=MARKER_LINEWIDTH,
                elinewidth=ERRORBAR_LINEWIDTH, capsize=ERRORBAR_CAPSIZE,
                capthick=ERRORBAR_LINEWIDTH, linestyle="None", zorder=1,
            )

        # Short split names ("Val." rather than "Validation"), as the +/-RMSE
        # legend already uses.  At 12 pt the full names make the legend wider
        # than the empty triangle above the parity line, so it would have to
        # cross the data whatever the placement scan does; the short ones fit.
        handles.append(Line2D(
            [0], [0], marker=style["marker"], linestyle="None", markersize=7,
            markerfacecolor=style["face"], markeredgecolor=style["edge"],
            label=bold(rf"$\mathrm{{{style['short']}}}$ ($R^2 = {score:.2f}$)"),
        ))

    values = np.concatenate(all_values)
    lims = [values.min() - PARITY_MARGIN, values.max() + PARITY_MARGIN]
    ax.plot(lims, lims, "k--", linewidth=1.4, zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(rf"$\mathrm{{Actual}}$ {TARGET_LABEL}",
                  fontsize=LABEL_FONTSIZE, labelpad=LABELPAD)
    ax.set_ylabel(rf"$\mathrm{{Predicted}}$ {TARGET_LABEL}",
                  fontsize=LABEL_FONTSIZE, labelpad=LABELPAD)
    style_axes(ax)
    ax.xaxis.set_major_locator(MultipleLocator(PARITY_TICK))
    ax.yaxis.set_major_locator(MultipleLocator(PARITY_TICK))
    return handles, plotted, ", ".join(scores)


def draw_residual(ax, splits: dict[str, tuple]) -> tuple[list, list, list, str]:
    """Residual vs predicted for the two evaluation splits, with +/-RMSE bands."""
    unit = unit_inner(TARGET_LABEL)
    split_handles, rmse_handles, scattered, scores = [], [], [], []

    for key in ("test", "val"):
        style = SPLIT_STYLE[key]
        y_true, y_pred, _ = splits[key]
        resid = y_true - y_pred
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        scattered.append(np.column_stack([y_pred, resid]))
        scores.append(f"{style['short']}={rmse:.3f}")

        ax.scatter(
            y_pred, resid,
            marker=style["marker"], s=MARKER_SIZE, alpha=MARKER_ALPHA,
            facecolors=style["face"], edgecolors=style["edge"],
            linewidths=MARKER_LINEWIDTH, zorder=1,
        )

        for sign in (+1, -1):
            ax.axhline(sign * rmse, color=style["edge"],
                       linewidth=BAND_LINEWIDTH, linestyle=style["band"],
                       alpha=0.95, zorder=3)

        split_handles.append(Line2D(
            [0], [0], marker=style["marker"], linestyle="None", markersize=7,
            markerfacecolor=style["face"], markeredgecolor=style["edge"],
            label=bold(style["label"]),
        ))

        rmse_txt = (rf"$\pm\mathrm{{RMSE}}_"
                    rf"{{\mathrm{{{style['short']}}}}} = {rmse:.2f}")
        rmse_txt += rf"\,/\,[{unit}]$" if RMSE_LEGEND_WITH_UNIT else r"$"
        rmse_handles.append(Line2D(
            [0], [0], color=style["edge"], linewidth=BAND_LINEWIDTH,
            linestyle=style["band"], label=bold(rmse_txt),
        ))

    ax.axhline(0, color="black", linewidth=1.3, linestyle="--", zorder=2)

    ax.set_xlabel(rf"$\mathrm{{Predicted}}$ {TARGET_LABEL}",
                  fontsize=LABEL_FONTSIZE, labelpad=LABELPAD)
    ax.set_ylabel(rf"$\mathrm{{Residual}}$ {TARGET_LABEL}",
                  fontsize=LABEL_FONTSIZE, labelpad=LABELPAD)
    style_axes(ax)
    ax.xaxis.set_major_locator(MultipleLocator(RESIDUAL_X_TICK))
    ax.yaxis.set_major_locator(MultipleLocator(RESIDUAL_Y_TICK))
    ax.set_ylim(RESIDUAL_Y_RANGE)
    return split_handles, rmse_handles, scattered, ", ".join(scores)


def build_panel(letter: str, dpi: int, outdirs: dict[bool, str],
                crop_each: bool = False,
                cache: dict[str, dict] | None = None) -> None:
    """Render one Figure 9 panel as PNG and PDF.

    See ``build_figure4.save_variants`` for how the two output sets relate:
    both come from one render and share a bounding box, so the legend-free
    copies overlay the manuscript ones exactly.
    """
    model_key, kind = PANELS[letter]
    model = MODELS[model_key]

    if cache is not None and model_key in cache:
        splits = cache[model_key]
    else:
        splits = model["load"]()
        if cache is not None:
            cache[model_key] = splits

    # No frame and no backing patch -- the labels sit directly on the panel,
    # as in Figure 4.
    legend_bg = dict(frameon=False, borderpad=0.25)

    fig, ax = ps.plot_init()

    if kind == "parity":
        handles, plotted, scores = draw_parity(ax, splits)
        # Axes geometry has to be final before any legend is trial-positioned.
        fig.tight_layout()
        placement, hits = place_parity_legend(
            fig, ax, handles, np.concatenate(plotted), legend_bg)
    else:
        split_handles, rmse_handles, scattered, scores = draw_residual(ax, splits)
        # Axes geometry has to be final before any legend is trial-positioned.
        fig.tight_layout()

        points = np.concatenate(scattered)
        if SHOW_SPLIT_LEGEND:
            split_legend = ax.legend(
                handles=split_handles,
                prop={"size": RMSE_FONTSIZE, "weight": "bold"},
                loc="lower center", ncol=2,
                handlelength=1.0, handletextpad=0.3, columnspacing=0.8,
                **legend_bg,
            )
            split_legend.set_zorder(6)
            ax.add_artist(split_legend)
            points = np.vstack([points,
                                legend_mask_points(fig, ax, split_legend)])

        placement, hits = place_rmse_legend(
            fig, ax, rmse_handles, points, legend_bg, letter)

    save_variants(fig, ax, outdirs, f"Figure9{letter}", dpi, crop_each)
    plt.close(fig)

    where = placement if hits is None else f"{placement} ({hits} behind)"
    print(f"  ({letter}) {model['name']:<5s} {kind:<9s} {scores:<40s} {where}")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--panels", default="all",
                        help="comma-separated panel letters, e.g. 'a,b' "
                             "(default: all 6)")
    parser.add_argument("--dpi", type=int, default=ps.resolution_value,
                        help=f"output resolution (default: {ps.resolution_value})")
    parser.add_argument("--out", default=OUTPUT_DIR,
                        help="parent output directory (default: PART_1/FIGURE9);"
                             f" panels land in {LEGEND_SUBDIR[True]}/ and "
                             f"{LEGEND_SUBDIR[False]}/")
    parser.add_argument("--legend", choices=("both", "on", "off"), default="both",
                        help="which set(s) to write (default: both)")
    parser.add_argument("--crop-each", action="store_true",
                        help="crop each set to its own content; by default both "
                             "share the legend version's bounding box so the "
                             "two sets are pixel-aligned")
    parser.add_argument("--overleaf", action="store_true",
                        help="also copy the with-legend PNGs over Figure9a..f.png"
                             f" in {OVERLEAF_DIR}")
    args = parser.parse_args(argv)

    if args.panels == "all":
        letters = list(PANELS)
    else:
        letters = [s.strip() for s in args.panels.split(",") if s.strip()]
        unknown = [s for s in letters if s not in PANELS]
        if unknown:
            parser.error(f"unknown panel(s): {', '.join(unknown)}")

    wanted = {"both": (True, False), "on": (True,), "off": (False,)}[args.legend]
    outdirs = {flag: os.path.join(args.out, LEGEND_SUBDIR[flag])
               for flag in wanted}

    if args.overleaf and True not in outdirs:
        parser.error("--overleaf needs the with-legend set (--legend both|on)")

    print(f"Figure 9: rebuilding {len(letters)} panel(s) at {args.dpi} dpi "
          f"-> {', '.join(LEGEND_SUBDIR[f] for f in wanted)}")
    cache: dict[str, dict] = {}   # each model feeds two panels; load once
    for letter in letters:
        build_panel(letter, args.dpi, outdirs, crop_each=args.crop_each,
                    cache=cache)
    for flag in wanted:
        print(f"wrote {outdirs[flag]}")

    if args.overleaf:
        # The Overleaf figures are gitignored, so keep a one-off copy of
        # whatever is being replaced before the first overwrite.
        backup = os.path.join(args.out, "published_backup")
        os.makedirs(backup, exist_ok=True)
        for letter in letters:
            name = f"Figure9{letter}.png"
            current = os.path.join(OVERLEAF_DIR, name)
            kept = os.path.join(backup, name)
            if os.path.exists(current) and not os.path.exists(kept):
                shutil.copy2(current, kept)
            shutil.copy2(os.path.join(outdirs[True], name), current)
        print(f"copied {len(letters)} PNG(s) to {OVERLEAF_DIR}")
        print(f"previous versions preserved in {backup}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
