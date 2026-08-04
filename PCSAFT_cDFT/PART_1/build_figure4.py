#!/usr/bin/env python
"""
build_figure4.py
================
Unified builder for every panel of Figure 4 (residual-vs-predicted analysis).

Figure 4 is a 4x4 grid: four target properties (rows) x four ML models
(columns).  Historically each of the 16 panels was produced by a different
notebook scattered over RF/, SVM/, XGBoost/ and TabPFN-snellius-home/, which
made a consistent restyling impossible.  This script rebuilds all 16 panels
from the stored ``*_predictions.csv`` files, so the whole figure is generated
by one command with one style definition.

Reviewer request (R?): improve colour-blind accessibility and enlarge the
legend text.  Both are addressed here:

* Colour is no longer the only channel carrying the Test/Validation
  distinction.  The two splits now also differ in marker shape (circle vs
  triangle) and in the line style of their +/-RMSE bands (dashed vs dash-dot),
  so the panels stay readable under dichromacy and in greyscale.  This
  redundancy, not the hue choice, is what fixes the accessibility problem:
  blue-vs-red is not a red-green confusion pair and survives protanopia,
  deuteranopia and tritanopia on its own.
* The +/-RMSE bands are drawn in the marker edge colours (luminance 0.152 and
  0.130) instead of `darkblue` and `darkred` (0.019 and 0.055).  Measured, this
  is a wash on the colour axis and slightly *worse* in greyscale -- it is the
  differing dash patterns, not the hue, that separate the two bands.  The
  previous bands merged because both were near-black *and* shared the same
  dash-dot style; that shared style was the real defect.
* Legend, axis-label and tick-label sizes are enlarged.  The panels are placed
  at 0.24\\textwidth in the manuscript, i.e. reduced by ~2.1-2.5x, so the 12 pt
  legend lands at ~4.8-5.9 pt on the page against ~4.7 pt before.
* The Test/Validation marker legend is dropped from the panels (see
  ``SHOW_SPLIT_LEGEND``); the marker/colour key is stated in the figure
  caption, and the +/-RMSE legend already shows both colours and line styles.
* The legend is placed in whichever band (top or bottom of the panel) the
  fewest markers fall behind; see ``scan_band``.  The full label with its unit,
  "+-RMSE_Test = 1.02 / [bar]", is 0.82 of the panel width at 12 pt (0.96 for
  the gamma row, whose unit is the longest), so every row fits inside the
  frame, though the widest leave little slack to dodge the markers with.  If
  the remaining overlaps ever have to go, a shorter label helps
  (``RMSE_LABEL_STYLE = "compact"``, 0.63 wide with the unit, 0.45 without), as
  does a legend outside the frame (``RMSE_LEGEND_PLACEMENT = "above"``).  The
  published wording is kept here by choice; the residual overlaps are touched
  up downstream, which is what the legend-free copies in ``no_legend/`` are
  for -- they share the legend version's bounding box, so the two sets overlay
  exactly.

Everything else -- data, splits, RMSE values, axis ranges, panel order -- is
unchanged with respect to the published version of the figure.

Usage
-----
    python build_figure4.py                 # -> FIGURE4/{with,no}_legend/Figure4a..p
    python build_figure4.py --overleaf      # also copy with_legend into OverleafDir
    python build_figure4.py --panels a,b,c  # rebuild a subset (quick iteration)
    python build_figure4.py --dpi 300       # draft resolution
    python build_figure4.py --legend off    # only the legend-free set
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

import thermoift.PLOT_SETTINGS as ps

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "FIGURE4")

# Two parallel sets are written, one per subdirectory: the panels as they go
# into the manuscript, and the same panels with the legend suppressed so the
# +/-RMSE labels can be re-laid by hand without fighting the rendered ones.
LEGEND_SUBDIR = {True: "with_legend", False: "no_legend"}
OVERLEAF_DIR = os.path.abspath(
    os.path.join(HERE, "..", "..", "OverleafDir", "A6-Draft-Overleaf")
)

# --------------------------------------------------------------------------- #
# Split style: colour + marker shape + band dash pattern (redundant encoding)   #
# --------------------------------------------------------------------------- #

SPLIT_STYLE = {
    "test": dict(
        label=r"$\mathrm{Test}$",
        short="Test",
        marker="o",                     # circle
        face="#56B4E9",                 # sky blue
        edge="#0072B2",                 # blue
        band=(0, (7, 3)),               # dashed
    ),
    "val": dict(
        label=r"$\mathrm{Validation}$",
        short="Val.",
        marker="^",                     # triangle -- shape carries the split too
        face="#FF6B6B",                 # red
        edge="#C1272D",                 # red, lighter than the previous darkred
        band=(0, (7, 2, 1.5, 2)),       # dash-dot
    ),
}

SHOW_SPLIT_LEGEND = False   # Test/Validation key lives in the caption instead

# Placement of the +/-RMSE legend.  Under "auto" every layout below is scored
# against the markers and the global minimum wins, ties going to the earlier
# (more compact) layout:
#
#   1. "row"   -- both entries on one line (two columns); shortest vertically
#                 but roughly twice as wide, so it never actually fits
#   2. "stack" -- the entries on two lines (one column); the usual choice
#   3. "split" -- one entry along the top band, the other along the bottom;
#                 halves the height, which rescues the densest panels
#
# Set to a Matplotlib loc string to pin the legend to one slot instead.
RMSE_LEGEND_LOC = "auto"
RMSE_LEGEND_LAYOUTS = ("row", "stack", "split")
LEGEND_PAD = 0.02           # gap between the legend box and the axes frame
LEGEND_CLEARANCE = 2.5      # extra clearance around the legend box, in points,
                            # so a marker may not even graze the frame
LEGEND_SCAN_STEPS = 161     # x positions tried along the top/bottom band
DEBUG_BOXES = bool(os.environ.get("FIG4_DEBUG_BOXES"))

# Font sizes (points, before the 0.24\textwidth reduction in the manuscript)
LEGEND_FONTSIZE = 17        # Test/Validation marker legend (if shown)
RMSE_FONTSIZE = 12          # +/-RMSE band legend
LABEL_FONTSIZE = 18         # x/y axis labels
TICK_LABELSIZE = 13         # major tick labels

RMSE_LEGEND_WITH_UNIT = True    # published form: "... = 1.02 / [bar]"

# "full"    -> "+-RMSE_Test = 1.02"   (0.65 of the panel width at 12 pt)
# "compact" -> "Test = 1.02"          (0.45), which is what leaves the legend
#              enough room to slide clear of the outlying markers.  The bands
#              are identified as +/-RMSE bands in the figure caption.
RMSE_LABEL_STYLE = os.environ.get("FIG4_RMSE_LABEL", "full")

# "inside" -> legend sits in the emptiest band of the axes (see the scan above)
# "above"  -> legend sits outside, over the frame: never collides, but costs
#             ~2 text lines of panel height
RMSE_LEGEND_PLACEMENT = os.environ.get("FIG4_LEGEND_PLACE", "inside")
MARKER_SIZE = 20
MARKER_ALPHA = 0.65
MARKER_LINEWIDTH = 0.6
BAND_LINEWIDTH = 1.6
LABELPAD = 4

# --------------------------------------------------------------------------- #
# Panel definitions                                                            #
# --------------------------------------------------------------------------- #

# LaTeX axis labels/symbols, matching thermoift.ML_postprocessing so the
# rebuilt panels are typographically identical to the published ones.
#
# ``y_tick``/``x_tick`` pin the major-tick spacing to the published figure.
# Matplotlib's automatic locator thins the ticks once the tick labels are
# enlarged (it fits ticks to the available label space), which would otherwise
# coarsen e.g. the P_bubble axis from steps of 5 to steps of 10.
TARGETS = {
    "p_bubble": dict(
        label=r"$P_{\mathrm{bubble}}$ / $[\mathrm{bar}]$",
        symbol=r"P_{\mathrm{bubble}}",
        y_range=(-18, 18), y_tick=10, x_tick=25,
    ),
    "p_dew": dict(
        label=r"$P_{\mathrm{dew}}$ / $[\mathrm{bar}]$",
        symbol=r"P_{\mathrm{dew}}",
        y_range=(-4, 4), y_tick=2, x_tick=20,
    ),
    "gamma": dict(
        label=r"$\gamma$ / $[\mathrm{mN\,m^{-1}}]$",
        symbol=r"\gamma",
        y_range=(-2, 2), y_tick=1, x_tick=5,
    ),
    "interfacial_thickness": dict(
        label=r"$L^{90}_{10}$ / $[\mathrm{nm}]$",
        symbol=r"L^{90}_{10}",
        y_range=(-1, 1), y_tick=0.5, x_tick=1,
    ),
}

# panel letter -> (target key, model name, predictions CSV relative to PART_1)
# Row order follows the manuscript: P_bubble, P_dew, gamma, thickness.
# Column order follows the manuscript: RF, SVR, XGBoost, TabPFN.
PANELS = {
    # --- P_bubble -----------------------------------------------------------
    "a": ("p_bubble", "RF",
          "RF/RF_Pbubble_OUTPUTS/RF_P_bubble_predictions.csv"),
    "b": ("p_bubble", "SVR",
          "SVM/SLURM_Pbubble/SVM_P_bubble_predictions.csv"),
    "c": ("p_bubble", "XGBoost",
          "XGBoost/XGBoost_Pbubble_OUTPUTS/RF_P_bubble_predictions.csv"),
    "d": ("p_bubble", "TabPFN",
          "TabPFN-snellius-home/without_HPO/TabPFN_P_bubble_OUTPUTS/"
          "TabPFN_P_bubble_predictions.csv"),
    # --- P_dew --------------------------------------------------------------
    "e": ("p_dew", "RF",
          "RF/RF_Pdew_OUTPUTS/RF_P_dew_predictions.csv"),
    "f": ("p_dew", "SVR",
          "SVM/SLURM_Pdew/SVM_P_dew_predictions.csv"),
    "g": ("p_dew", "XGBoost",
          "XGBoost/XGBoost_Pdew_OUTPUTS/XGB_P_dew_predictions.csv"),
    "h": ("p_dew", "TabPFN",
          "TabPFN-snellius-home/without_HPO/TabPFN_P_dew_OUTPUTS/"
          "TabPFN_P_dew_predictions.csv"),
    # --- gamma --------------------------------------------------------------
    "i": ("gamma", "RF",
          "RF/RF_GAMMA_OUTPUTS/RF_gamma_predictions.csv"),
    "j": ("gamma", "SVR",
          "SVM/SLURM_gamma/SVM_gamma_predictions.csv"),
    "k": ("gamma", "XGBoost",
          "XGBoost/XGBoost_GAMMA_OUTPUTS/RF_gamma_predictions.csv"),
    "l": ("gamma", "TabPFN",
          "TabPFN-snellius-home/without_HPO/TabPFN_gamma_OUTPUTS/"
          "TabPFN_gamma_predictions.csv"),
    # --- interfacial thickness ----------------------------------------------
    "m": ("interfacial_thickness", "RF",
          "RF/RF_THICKNESS_OUTPUTS/RF_interfacial_thickness_predictions.csv"),
    "n": ("interfacial_thickness", "SVR",
          "SVM/SLURM_thickness/SVM_interfacial_thickness_predictions.csv"),
    "o": ("interfacial_thickness", "XGBoost",
          "XGBoost/XGBoost_THICKNESS_OUTPUTS/"
          "XGBoost_interfacial_thickness_predictions.csv"),
    "p": ("interfacial_thickness", "TabPFN",
          "TabPFN-snellius-home/without_HPO/TabPFN_interfacial_thickness_OUTPUTS/"
          "TabPFN_interfacial_thickness_predictions.csv"),
}


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def bold(s: str) -> str:
    """Bold a legend label, including its math content, under text.usetex."""
    return rf"{{\boldmath\bfseries {s}}}"


def unit_inner(label: str) -> str:
    r"""``$P$ / $[\mathrm{bar}]$`` -> ``\mathrm{bar}`` (bare unit for legends)."""
    unit = label.split("/")[-1].strip()
    if unit.startswith("$[") and unit.endswith("]$"):
        return unit.strip("$")[1:-1]
    return unit


def load_splits(csv_path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{split: (y_true, y_pred)}`` for the test and validation splits."""
    df = pd.read_csv(csv_path)
    return {
        key: (
            df.loc[df["split"] == key, "actual"].to_numpy(),
            df.loc[df["split"] == key, "predicted"].to_numpy(),
        )
        for key in ("test", "val")
        if (df["split"] == key).any()
    }


def visible_marker_frac(ax, points: np.ndarray) -> np.ndarray:
    """Marker centres in axes-fraction coordinates, clipped to the axes."""
    frac = ax.transAxes.inverted().transform(ax.transData.transform(points))
    return frac[(frac[:, 0] >= 0) & (frac[:, 0] <= 1)
                & (frac[:, 1] >= 0) & (frac[:, 1] <= 1)]


def legend_box(fig, ax, legend) -> tuple[tuple, tuple]:
    """Rendered legend size in axes fractions: ``(raw, inflated)``.

    The inflated box adds the marker radius plus a little slack, so a marker
    counts as a collision when its glyph touches the legend even though its
    centre lies outside the frame.
    """
    fig.canvas.draw()
    bb = legend.get_window_extent(fig.canvas.get_renderer())
    inv = ax.transAxes.inverted()
    pad = (LEGEND_CLEARANCE + math.sqrt(MARKER_SIZE) / 2.0) * fig.dpi / 72.0
    (rx0, ry0), (rx1, ry1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    (px0, py0), (px1, py1) = inv.transform([[bb.x0 - pad, bb.y0 - pad],
                                            [bb.x1 + pad, bb.y1 + pad]])
    return (rx1 - rx0, ry1 - ry0), (px1 - px0, py1 - py0)


def markers_behind(frac: np.ndarray, x0, y0, w, h) -> int:
    """Number of visible markers inside the box with corner ``(x0, y0)``."""
    return int(np.count_nonzero(
        (frac[:, 0] >= x0) & (frac[:, 0] <= x0 + w)
        & (frac[:, 1] >= y0) & (frac[:, 1] <= y0 + h)
    ))


def scan_band(frac: np.ndarray, w: float, h: float, vert: str):
    """Slide a ``w`` x ``h`` legend along the top or bottom edge of the axes.

    Three fixed slots (left/centre/right) are too coarse for these panels: the
    residual clouds leave a clear band above and below the data, but the gap is
    rarely centred.  Sweeping the box across the band instead finds the
    genuinely empty x, and ties break towards the middle of the panel.
    """
    span = 1.0 - 2 * LEGEND_PAD - w
    y0 = LEGEND_PAD if vert == "lower" else 1.0 - LEGEND_PAD - h
    if span <= 0:
        # Wider than the axes -- the long units do this.  Centre it and report
        # the overflow, which is scored ahead of the marker count so that a
        # legend which fits always beats one that hangs off the panel.
        x0 = (1.0 - w) / 2.0
        return -span, markers_behind(frac, x0, y0, w, h), x0, y0
    best = None
    for x0 in np.linspace(LEGEND_PAD, LEGEND_PAD + span, LEGEND_SCAN_STEPS):
        key = (markers_behind(frac, x0, y0, w, h), abs(x0 + w / 2 - 0.5))
        if best is None or key < best[0]:
            best = (key, float(x0))
    return 0.0, best[0][0], best[1], y0


def place_rmse_legend(fig, ax, handles, points: np.ndarray,
                      legend_bg: dict, tag: str = "") -> tuple[str, int | None]:
    """Draw the +/-RMSE legend in the emptiest band of the axes.

    ``points`` are the plotted ``(x, y)`` pairs in data coordinates; every
    candidate position is scored against them and the one hiding the fewest
    markers wins.  Returns a short description of where the legend ended up
    and how many markers remain behind it (``None`` when ``RMSE_LEGEND_LOC``
    pinned the position instead of searching for one).

    Shared with ``build_figure9.py`` so both figures place their legends by
    the same rules.
    """
    def make_legend(handles, ncol, anchor=None):
        """Draw the legend, anchored by its lower-left corner when given."""
        kw = dict(loc="lower left", bbox_to_anchor=anchor,
                  bbox_transform=ax.transAxes) if anchor else dict(loc="best")
        leg = ax.legend(
            handles=handles,
            prop={"size": RMSE_FONTSIZE, "weight": "bold"},
            ncol=ncol,
            handlelength=1.8, handletextpad=0.5, columnspacing=1.2,
            **kw, **legend_bg,
        )
        leg.set_zorder(6)
        return leg

    frac = visible_marker_frac(ax, points)

    def probe(handles, ncol):
        """Measure a legend without keeping it, then sweep both bands."""
        p = make_legend(handles, ncol)
        raw, infl = legend_box(fig, ax, p)
        p.remove()
        found = [(scan_band(frac, *infl, vert), raw, infl)
                 for vert in ("lower", "upper")]
        return [(f, raw, infl) for f, raw, infl in found if f is not None]

    def anchor_for(x0, y0, raw, infl):
        """Lower-left anchor of the drawn frame inside its inflated box."""
        return (x0 + (infl[0] - raw[0]) / 2.0, y0 + (infl[1] - raw[1]) / 2.0)

    if RMSE_LEGEND_PLACEMENT == "above":
        # Outside the frame entirely: no marker can ever be obscured, at the
        # cost of ~2 text lines of height above each panel.
        leg = ax.legend(
            handles=handles,
            prop={"size": RMSE_FONTSIZE, "weight": "bold"},
            loc="lower center", bbox_to_anchor=(0.5, 1.005),
            bbox_transform=ax.transAxes, ncol=1,
            handlelength=1.8, handletextpad=0.5, columnspacing=1.2,
            frameon=False,
        )
        leg.set_zorder(6)
        return "above axes", 0

    if RMSE_LEGEND_LOC != "auto":
        make_legend(handles, 1)
        return RMSE_LEGEND_LOC, None

    # Score every layout across both bands, then take the global minimum.
    # RMSE_LEGEND_LAYOUTS breaks ties, so the most compact legend wins
    # whenever it is no worse than a taller or a split one.
    options = []
    for rank, layout in enumerate(RMSE_LEGEND_LAYOUTS):
        if layout == "split":
            # One entry per line, one along the top band and one along the
            # bottom: two minimal boxes, each free to find its own gap.
            tops = [o for o in probe([handles[0]], 1) if o[0][3] > 0.5]
            bots = [o for o in probe([handles[1]], 1) if o[0][3] < 0.5]
            if not tops or not bots:
                continue
            (to, tn, tx, ty), traw, tinfl = min(tops)
            (bo, bn, bx, by), braw, binfl = min(bots)
            options.append((max(to, bo), tn + bn, rank, layout,
                            [(anchor_for(tx, ty, traw, tinfl), [handles[0]]),
                             (anchor_for(bx, by, braw, binfl), [handles[1]])]))
        else:
            ncol = 2 if layout == "row" else 1
            found = probe(handles, ncol)
            if not found:
                continue
            (over, n, x0, y0), raw, infl = min(found)
            options.append((over, n, rank, layout,
                            [(anchor_for(x0, y0, raw, infl), handles)]))

    if DEBUG_BOXES:
        print(f"      options for ({tag}): "
              + ", ".join(f"{lay}={n}(ovf {o:.2f})"
                          for o, n, _, lay, _ in options))

    if not options:
        # Every layout is wider than the axes -- typically a long unit in
        # the label.  Fall back to placing the legend outside the frame,
        # which has no width limit, rather than failing the build.
        leg = ax.legend(
            handles=handles,
            prop={"size": RMSE_FONTSIZE, "weight": "bold"},
            loc="lower center", bbox_to_anchor=(0.5, 1.005),
            bbox_transform=ax.transAxes, ncol=1,
            handlelength=1.8, handletextpad=0.5, frameon=False,
        )
        leg.set_zorder(6)
        placement, hits = "above axes (no in-panel fit)", 0
    else:
        over, hits, _, layout, placements = min(options)
        ncol = 2 if layout == "row" else 1
        for anchor, hs in placements[:-1]:
            ax.add_artist(make_legend(hs, ncol, anchor))
        make_legend(placements[-1][1], ncol, placements[-1][0])
        placement = layout + "".join(
            f" @({a[0]:.2f},{a[1]:.2f})" for a, _ in placements
        ) + (f" [overflows {over:.2f}]" if over > 0 else "")

    if DEBUG_BOXES:
        for leg in [a for a in ax.get_children()
                    if isinstance(a, matplotlib.legend.Legend)]:
            (rw, rh), (iw, ih) = legend_box(fig, ax, leg)
            bb = leg.get_window_extent(fig.canvas.get_renderer())
            (x0, y0), _ = ax.transAxes.inverted().transform(
                [[bb.x0, bb.y0], [bb.x1, bb.y1]])
            ax.add_patch(Rectangle(
                (x0 - (iw - rw) / 2, y0 - (ih - rh) / 2), iw, ih,
                transform=ax.transAxes, facecolor="none",
                edgecolor="lime", linewidth=1.2, zorder=10))

    return placement, hits


def save_variants(fig, ax, outdirs: dict[bool, str], stem: str, dpi: int,
                  crop_each: bool = False) -> None:
    """Save ``stem``.png/.pdf with the legend drawn, hidden, or both.

    ``outdirs`` maps ``True``/``False`` (legend drawn / hidden) to an output
    directory; a missing key skips that variant.  Both variants come from a
    single render, and by default both are cropped to the *same* bounding box
    -- the one the legend version needs.  That costs the legend-free panels
    some outer whitespace, but it makes the two sets pixel-aligned, so they can
    be stacked in Illustrator and the axes coincide exactly.  Pass
    ``crop_each`` to crop each variant to its own content instead.

    Shared with ``build_figure9.py``.
    """
    def save(outdir: str, bbox) -> None:
        os.makedirs(outdir, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(outdir, f"{stem}.{ext}"),
                        dpi=dpi, bbox_inches=bbox)

    # Reference crop: what "tight" would give for the legend version.  The
    # overflowing legend widens it, which is exactly why it has to be measured
    # once and reused for the legend-free variant.
    pad = plt.rcParams["savefig.pad_inches"]
    fig.canvas.draw()
    shared = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad)

    if outdirs.get(True):
        save(outdirs[True], shared)

    if outdirs.get(False):
        for artist in ax.get_children():
            if isinstance(artist, matplotlib.legend.Legend):
                artist.set_visible(False)
        if crop_each:
            fig.canvas.draw()
            bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad)
        else:
            bbox = shared
        save(outdirs[False], bbox)


def build_panel(letter: str, dpi: int, outdirs: dict[bool, str],
                crop_each: bool = False) -> None:
    """Render one residual-vs-predicted panel as PNG and PDF.

    See :func:`save_variants` for how the two output sets relate.
    """
    target_key, model, rel_csv = PANELS[letter]
    target = TARGETS[target_key]

    csv_path = os.path.join(HERE, rel_csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"panel ({letter}): missing predictions {csv_path}")

    splits = load_splits(csv_path)
    unit = unit_inner(target["label"])

    fig, ax = ps.plot_init()

    # zero-residual reference
    ax.axhline(0, color="black", linewidth=1.3, linestyle="--", zorder=2)

    split_handles, rmse_handles, scattered = [], [], []

    for key, (y_true, y_pred) in splits.items():
        style = SPLIT_STYLE[key]
        resid = y_true - y_pred
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        scattered.append(np.column_stack([y_pred, resid]))

        ax.scatter(
            y_pred, resid,
            marker=style["marker"],
            s=MARKER_SIZE,
            alpha=MARKER_ALPHA,
            facecolors=style["face"],
            edgecolors=style["edge"],
            linewidths=MARKER_LINEWIDTH,
            zorder=1,
        )

        for sign in (+1, -1):
            ax.axhline(
                sign * rmse,
                color=style["edge"],
                linewidth=BAND_LINEWIDTH,
                linestyle=style["band"],
                alpha=0.95,
                zorder=3,
            )

        split_handles.append(
            Line2D(
                [0], [0],
                marker=style["marker"], linestyle="None", markersize=7,
                markerfacecolor=style["face"], markeredgecolor=style["edge"],
                label=bold(style["label"]),
            )
        )

        if RMSE_LABEL_STYLE == "compact":
            rmse_txt = rf"$\mathrm{{{style['short']}}} = {rmse:.2f}"
        else:
            rmse_txt = (rf"$\pm\mathrm{{RMSE}}_"
                        rf"{{\mathrm{{{style['short']}}}}} = {rmse:.2f}")
        rmse_txt += rf"\,/\,[{unit}]$" if RMSE_LEGEND_WITH_UNIT else r"$"
        rmse_handles.append(
            Line2D(
                [0], [0],
                color=style["edge"], linewidth=BAND_LINEWIDTH,
                linestyle=style["band"], label=bold(rmse_txt),
            )
        )

    ax.set_xlabel(rf"$\mathrm{{Predicted}}$ {target['label']}",
                  fontsize=LABEL_FONTSIZE, labelpad=LABELPAD)
    ax.set_ylabel(rf"$\mathrm{{Residual}}$ ${target['symbol']}$ / "
                  rf"${target['label'].split('/')[-1].strip().strip('$')}$",
                  fontsize=LABEL_FONTSIZE, labelpad=LABELPAD)

    ps.apply_axis_style(ax)

    # No frame and no backing patch -- the labels sit directly on the panel.
    # The placement scan below is what keeps them clear of the markers; where a
    # marker does fall behind a label it now overprints it outright, since
    # there is no longer a translucent patch to soften the overlap.
    legend_bg = dict(frameon=False, borderpad=0.25)

    # Optional legend: Test / Validation markers, two columns, bottom centre.
    # Off by default -- the marker/colour key is given in the figure caption,
    # and the +/-RMSE legend below already shows both colours and line styles.
    if SHOW_SPLIT_LEGEND:
        split_legend = ax.legend(
            handles=split_handles,
            prop={"size": LEGEND_FONTSIZE, "weight": "bold"},
            loc="lower center", ncol=2,
            handlelength=1.0, handletextpad=0.3, columnspacing=0.8,
            **legend_bg,
        )
        split_legend.set_zorder(6)
        ax.add_artist(split_legend)

    ax.minorticks_on()
    ax.xaxis.set_major_locator(MultipleLocator(target["x_tick"]))
    ax.yaxis.set_major_locator(MultipleLocator(target["y_tick"]))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)
    ax.tick_params(axis="both", which="minor", length=3)
    ax.set_ylim(target["y_range"])

    # Axes geometry has to be final before the legend can be trial-positioned.
    fig.tight_layout()

    placement, hits = place_rmse_legend(
        fig, ax, rmse_handles, np.concatenate(scattered), legend_bg, letter)

    save_variants(fig, ax, outdirs, f"Figure4{letter}", dpi, crop_each)
    plt.close(fig)

    rmses = ", ".join(
        f"{SPLIT_STYLE[k]['short']}={np.sqrt(np.mean((t - p) ** 2)):.3f}"
        for k, (t, p) in splits.items()
    )
    where = placement if hits is None else f"{placement} ({hits} behind)"
    print(f"  ({letter}) {model:<8s} {target_key:<22s} {rmses:<26s} {where}")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--panels", default="all",
                        help="comma-separated panel letters, e.g. 'a,e,i' "
                             "(default: all 16)")
    parser.add_argument("--dpi", type=int, default=ps.resolution_value,
                        help=f"output resolution (default: {ps.resolution_value})")
    parser.add_argument("--out", default=OUTPUT_DIR,
                        help="parent output directory (default: PART_1/FIGURE4);"
                             f" panels land in {LEGEND_SUBDIR[True]}/ and "
                             f"{LEGEND_SUBDIR[False]}/")
    parser.add_argument("--legend", choices=("both", "on", "off"), default="both",
                        help="which set(s) to write (default: both)")
    parser.add_argument("--crop-each", action="store_true",
                        help="crop each set to its own content; by default both "
                             "share the legend version's bounding box so the "
                             "two sets are pixel-aligned")
    parser.add_argument("--overleaf", action="store_true",
                        help="also copy the with-legend PNGs over Figure4a..p.png"
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

    print(f"Figure 4: rebuilding {len(letters)} panel(s) at {args.dpi} dpi "
          f"-> {', '.join(LEGEND_SUBDIR[f] for f in wanted)}")
    for letter in letters:
        build_panel(letter, args.dpi, outdirs, crop_each=args.crop_each)
    for flag in wanted:
        print(f"wrote {outdirs[flag]}")

    if args.overleaf and True not in outdirs:
        parser.error("--overleaf needs the with-legend set (--legend both|on)")

    if args.overleaf:
        # The Overleaf figures are gitignored, so keep a one-off copy of
        # whatever is being replaced before the first overwrite.
        backup = os.path.join(args.out, "published_backup")
        os.makedirs(backup, exist_ok=True)
        for letter in letters:
            name = f"Figure4{letter}.png"
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
