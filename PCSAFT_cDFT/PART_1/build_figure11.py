#!/usr/bin/env python
"""
build_figure11.py
=================
Builder for the prediction-interval figures: SI Figure S26 (TabPFN, P_bubble)
and SI Figure S27 (WSD + SVGP, IFT), drawn locally as Figure11a-c and
Figure12a-c.

Reviewer request (R2): "It would be helpful to include one example showing
95 % prediction intervals from TabPFN to illustrate model confidence."

Two models, because only one of them can answer that request directly:

* TabPFN predicts a distribution, not a number.  The point predictions used
  everywhere else in the manuscript are the means of a discretised predictive
  density, and the 2.5 % / 97.5 % quantiles of that same density bound the
  central 95 % interval.  That holds for the archived P_bubble model, which is
  a plain ``TabPFNRegressor``.
* The archived gamma model is a ``DecisionTreeTabPFNRegressor``, whose output
  is an aggregate of the TabPFNs fitted at each tree node and which exposes no
  quantiles.  The IFT intervals therefore come from the hybrid WSD + SVGP
  residual model of Section 3.6, a GP and so probabilistic by construction.

``tabpfn_prediction_intervals.py`` and ``svgp_prediction_intervals.py`` do the
inference and write FIGURE11/; this script only draws, so the figures can be
restyled without paying for inference again.

Three panels each:

(a) The example feed against T, with the PCP-SAFT EoS + cDFT reference, the
    model mean at each held-out state point, and the 95 % interval.  For
    P_bubble the interval is a shaded band and it is narrow on the scale of the
    curve -- which is the point of the panel, and the reason for (b).  gamma
    depends on P as well, so its panel is drawn on the bubble-pressure slice
    and the interval is drawn per point.
(b) The same points as a deviation, model mean minus reference, with the 95 %
    interval as error bars.  This is where the interval is legible and where
    one can see the reference falling inside it.
(c) Whether those intervals mean what they say: empirical against nominal
    coverage of the central intervals on the whole test split.

Style, fonts and the saving machinery are Figure 4's, imported from
``build_figure4`` so the figures cannot drift apart.  Test and validation keep
Figure 4's redundant encoding -- blue circle / red triangle, so the two splits
stay separable under dichromacy and in greyscale.

Usage
-----
    python build_figure11.py                       # -> FIGURE11/{with,no}_legend/Figure11a..c
    python build_figure11.py --model svgp          # the IFT figure, Figure12a..c
    python build_figure11.py --overleaf            # also copy into the SI as FigureS26/S27a..c
    python build_figure11.py --feed 3              # draw a different Table 1 feed
    python build_figure11.py --panels a,b          # rebuild a subset
    python build_figure11.py --dpi 300             # draft resolution
"""

from __future__ import annotations

import argparse
import json
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

# Style, fonts and the saving machinery are Figure 4's.
from build_figure4 import (
    apply_font_settings,
    LABELPAD,
    LABEL_FONTSIZE,
    LEGEND_SUBDIR,
    MARKER_ALPHA,
    MARKER_LINEWIDTH,
    MARKER_SIZE,
    RMSE_FONTSIZE,
    SPLIT_STYLE,
    TICK_LABELSIZE,
    bold,
    save_variants,
)

# See build_figure4 for why this is repeated rather than left to the import.
apply_font_settings()

# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "FIGURE11")
DATASET = os.path.join(HERE, "interfacial_results_dataset_A4.csv")
RESIDUAL_DATASET = os.path.join(HERE, "RESIDUAL", "CombinedDatasetSEC_A4.csv")
# The panels live in the Supporting Information (Section S4.3), not the main
# manuscript, so they are copied out as FigureS26/S27a-c.
OVERLEAF_DIR = os.path.abspath(
    os.path.join(HERE, "..", "..", "OverleafDir", "Supporting-Information-A6")
)

# --------------------------------------------------------------------------- #
# Panel settings                                                               #
# --------------------------------------------------------------------------- #

# Which feed of Table 1 carries panels (a) and (b).  Feed 2 is the four
# component feed with the largest H2 content of the table, and z_H2 is the
# feature the SHAP analysis ranks first for P_bubble, so it is the feed whose
# bubble curve is hardest to learn -- the honest choice for an example, and the
# one where the low-temperature widening discussed in Section 3.4 should show
# up if it shows up anywhere.
DEFAULT_FEED = 2

LEVEL = 0.95                    # the interval the reviewer asked for
BAND_FACE = "#9ECAE1"           # 95 % band fill, same hue family as Test
BAND_ALPHA = 0.55
REF_COLOR = "black"
REF_LINEWIDTH = 1.3
ERRORBAR_LINEWIDTH = 0.9
ERRORBAR_CAPSIZE = 2.0
DIAGONAL_STYLE = dict(color="black", linewidth=1.3, linestyle="--")

# Axis labels are kept short on purpose.  Under the pgf backend Matplotlib
# cannot measure typeset LaTeX text, so a long rotated y-label is not accounted
# for by the tight bounding box and comes out clipped; the definitions live in
# the caption instead.
T_LABEL = r"$T$ / $[\mathrm{K}]$"
NOMINAL_LABEL = r"$\mathrm{Nominal\ level}$ / $[\%]$"
EMPIRICAL_LABEL = r"$\mathrm{Coverage}$ / $[\%]$"

PANELS = ("a", "b", "c")

# The two probabilistic models of the paper, and how each stores its interval.
#
# TabPFN emits a density over the target, so the CSV carries a quantile ladder
# and the interval is a pair of quantile columns.  The SVGP is a Gaussian
# process on the WSD residual: its interval was formed in the transformed space
# and mapped back, so the CSV carries the endpoints directly.  Everything below
# is driven off these two entries.
MODELS = {
    "tabpfn": {
        "intervals": "TabPFN_P_bubble_intervals.csv",
        "coverage": "TabPFN_P_bubble_coverage.json",
        "source": "tabpfn_prediction_intervals.py",
        "stem": "Figure11",
        "overleaf_stem": "FigureS26",
        "series": r"\mathrm{TabPFN}",
        "value_label": r"$P_{\mathrm{bubble}}$ / $[\mathrm{bar}]$",
        "dev_label": r"$\Delta P_{\mathrm{bubble}}$ / $[\mathrm{bar}]$",
        "reference_label": r"$\mathrm{PCP\text{-}SAFT\ EoS} + \mathrm{cDFT}$",
        "bounds": lambda lvl: (f"q{(1 - lvl) / 2:.4f}", f"q{(1 + lvl) / 2:.4f}"),
        "dataset": DATASET,
        "reference_column": "P_bubble",
        "temperature_column": "temperature",
        "pressure_column": "pressure",
        # P_bubble is a property of (T, z) alone, so the state points a feed has
        # at one temperature share one reference value, and every state point of
        # the feed sits on the one curve.  A held-out point at nearly every
        # temperature then supports a filled band.
        "reference_reduction": "first",
        "slice_rank": None,
        "band": True,
        # Both panels fall steeply and then flatten, leaving the top right free.
        "legend_loc": {"a": "upper right", "b": "upper right"},
    },
    "svgp": {
        "intervals": "SVGP_gamma_intervals.csv",
        "coverage": "SVGP_gamma_coverage.json",
        "source": "svgp_prediction_intervals.py",
        "stem": "Figure12",
        "overleaf_stem": "FigureS27",
        "series": r"\mathrm{WSD}+\mathrm{SVGP}",
        "value_label": r"$\gamma$ / $[\mathrm{mN\,m^{-1}}]$",
        # Delta-gamma already denotes the cDFT - WSD residual in Fig. 10,
        # so the deviation of the reconstructed IFT gets its own symbol.
        "dev_label": r"$\delta \gamma$ / $[\mathrm{mN\,m^{-1}}]$",
        "reference_label": r"$\mathrm{PCP\text{-}SAFT\ EoS} + \mathrm{cDFT}$",
        "reference_label_short": r"$\mathrm{cDFT}$",
        "bounds": lambda lvl: (f"lo{lvl:.2f}", f"hi{lvl:.2f}"),
        "dataset": RESIDUAL_DATASET,
        "reference_column": "gamma_cDFT_UC",
        "temperature_column": "T",
        "pressure_column": "P",
        # gamma depends on P as well as on (T, z) -- strongly so at low T, where
        # it falls by ca. 4 mN/m across the pressure grid -- so gamma against T
        # alone is not a curve, and a band over T would fold that physical
        # spread into what should read as model uncertainty.  Each feed is
        # sampled on a five-point grid running from the dew to the bubble
        # pressure at every temperature, so panel (a) takes the last of those,
        # i.e. gamma along the bubble curve, which is the slice Figure S26 is
        # drawn on.  Too few held-out points survive that cut to interpolate a
        # band, so the interval is drawn per point instead.
        "reference_reduction": "first",
        "slice_rank": 5,
        "band": False,
        # gamma runs from ca. 17 mN/m down to 0 across the temperature range, so
        # panel (a) is free at the bottom left rather than the top right.
        "legend_loc": {"a": "lower left", "b": "upper right"},
    },
}


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #

def load_intervals(data_dir: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    paths = [os.path.join(data_dir, cfg[key])
             for key in ("intervals", "coverage")]
    for path in paths:
        if not os.path.exists(path):
            raise SystemExit(f"{path} is missing; run {cfg['source']} first")
    frame = pd.read_csv(paths[0], index_col="idx")
    with open(paths[1]) as fh:
        summary = json.load(fh)
    return frame, summary


def feed_rows(frame: pd.DataFrame, summary: dict, feed: int) -> pd.DataFrame:
    """Held-out state points of one Table 1 feed, ordered by T then P."""
    mapping = {int(k): v for k, v in summary["table1_feed_to_source_id"].items()}
    if feed not in mapping:
        raise SystemExit(
            f"Table 1 feed {feed} is not in the dataset; available: "
            f"{sorted(mapping)}")
    rows = frame[frame.source_id == mapping[feed]]
    return rows.sort_values(["temperature", "pressure"])


def feed_frame(source_id: int, cfg: dict) -> pd.DataFrame:
    """All state points of a feed, from the dataset the model was trained on.

    Carries a ``rank`` column: the position of each point on the pressure grid
    its temperature is sampled on, 1 at the dew pressure up to the grid size at
    the bubble pressure.
    """
    df = pd.read_csv(cfg["dataset"])
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    t_col, p_col, y_col = (cfg["temperature_column"], cfg["pressure_column"],
                           cfg["reference_column"])
    if y_col not in df.columns:            # the residual set stores it in parts
        df[y_col] = df["gamma_wsd_UC"] + df["gamma_cDFT_minus_wsd_uncorrected"]
    rows = df[df.source_id == source_id].copy()
    rows["rank"] = (rows.groupby(t_col)[p_col]
                    .rank(method="first").astype(int))
    return rows.rename(columns={t_col: "temperature", p_col: "pressure",
                                y_col: "value"})


def slice_of(rows: pd.DataFrame, feed: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Held-out points restricted to the pressure slice panel (a) is drawn on.

    Matched on rounded (T, P) so the join does not depend on the two CSVs
    round-tripping their floats identically.
    """
    if cfg["slice_rank"] is None:
        return rows
    key = lambda f: list(zip(f["temperature"].round(6), f["pressure"].round(6)))
    keep = set(key(feed[feed["rank"] == cfg["slice_rank"]]))
    return rows[[k in keep for k in key(rows)]]


def reference_curve(feed: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The cDFT reference against temperature, on panel (a)'s pressure slice."""
    rows = feed if cfg["slice_rank"] is None else feed[feed["rank"] == cfg["slice_rank"]]
    return (rows.groupby("temperature")["value"].agg(cfg["reference_reduction"])
            .reset_index().sort_values("temperature"))


def split_frames(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Rows grouped by split, in Figure 4's order, skipping empty splits."""
    return {key: rows[rows.split == key]
            for key in ("test", "val") if (rows.split == key).any()}


# --------------------------------------------------------------------------- #
# Panels                                                                       #
# --------------------------------------------------------------------------- #

def style_axes(ax) -> None:
    ps.apply_axis_style(ax)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)
    ax.tick_params(axis="both", which="minor", length=3)


def marker_handle(style: dict, label: str) -> Line2D:
    return Line2D(
        [0], [0], marker=style["marker"], linestyle="None", markersize=7,
        markerfacecolor=style["face"], markeredgecolor=style["edge"],
        label=bold(label),
    )


def draw_curve(ax, rows: pd.DataFrame, curve: pd.DataFrame, cfg: dict) -> list:
    """Panel (a): the reference curve, the mean, and the 95 % band."""
    lo, hi = cfg["bounds"](LEVEL)

    if cfg["band"]:
        # The band is drawn over the temperature axis, so the several state
        # points a feed has at one temperature are collapsed to the widest
        # interval seen there: the envelope of the predictions, not an average.
        by_t = rows.groupby("temperature").agg(lo=(lo, "min"), hi=(hi, "max"))
        ax.fill_between(by_t.index, by_t["lo"], by_t["hi"],
                        facecolor=BAND_FACE, alpha=BAND_ALPHA, linewidth=0,
                        zorder=2)

    ax.plot(curve["temperature"], curve["value"],
            color=REF_COLOR, linewidth=REF_LINEWIDTH, zorder=3)

    handles = [
        Line2D([0], [0], color=REF_COLOR, linewidth=REF_LINEWIDTH,
               label=bold(cfg["reference_label"] if cfg["band"]
                          else cfg["reference_label_short"])),
    ]
    for key, part in split_frames(rows).items():
        style = SPLIT_STYLE[key]
        if cfg["band"]:
            ax.scatter(part["temperature"], part["mean"],
                       marker=style["marker"], s=MARKER_SIZE, alpha=MARKER_ALPHA,
                       facecolors=style["face"], edgecolors=style["edge"],
                       linewidths=MARKER_LINEWIDTH, zorder=4)
        else:
            ax.errorbar(
                part["temperature"], part["mean"],
                yerr=np.vstack([part["mean"] - part[lo],
                                part[hi] - part["mean"]]),
                fmt=style["marker"], markersize=np.sqrt(MARKER_SIZE),
                markerfacecolor=style["face"], markeredgecolor=style["edge"],
                markeredgewidth=MARKER_LINEWIDTH,
                ecolor=style["edge"], elinewidth=ERRORBAR_LINEWIDTH,
                capsize=ERRORBAR_CAPSIZE, capthick=ERRORBAR_LINEWIDTH,
                linestyle="None", alpha=0.9, zorder=4,
            )
        # With a band there is room to name the model in the legend.  With error
        # bars the free corner is smaller, so the labels stay bare and the
        # caption carries the model name and the meaning of the bars.
        handles.append(marker_handle(
            style, rf"${cfg['series']}$ ({style['short']})" if cfg["band"]
            else style["short"]))
    if cfg["band"]:
        handles.append(Line2D(
            [0], [0], marker="s", linestyle="None", markersize=8,
            markerfacecolor=BAND_FACE, markeredgecolor="none", alpha=BAND_ALPHA,
            label=bold(rf"${int(LEVEL * 100)}\,\%\ \mathrm{{PI}}$")))

    ax.set_xlabel(T_LABEL, fontsize=LABEL_FONTSIZE, labelpad=LABELPAD)
    ax.set_ylabel(cfg["value_label"], fontsize=LABEL_FONTSIZE, labelpad=LABELPAD)
    return handles


def draw_deviation(ax, rows: pd.DataFrame, cfg: dict) -> list:
    """Panel (b): mean minus reference, with the 95 % interval as error bars."""
    lo, hi = cfg["bounds"](LEVEL)
    ax.axhline(0, **DIAGONAL_STYLE, zorder=2)

    handles = []
    for key, part in split_frames(rows).items():
        style = SPLIT_STYLE[key]
        centre = part["mean"] - part["actual"]
        ax.errorbar(
            part["temperature"], centre,
            yerr=np.vstack([centre - (part[lo] - part["actual"]),
                            (part[hi] - part["actual"]) - centre]),
            fmt=style["marker"], markersize=np.sqrt(MARKER_SIZE),
            markerfacecolor=style["face"], markeredgecolor=style["edge"],
            markeredgewidth=MARKER_LINEWIDTH,
            ecolor=style["edge"], elinewidth=ERRORBAR_LINEWIDTH,
            capsize=ERRORBAR_CAPSIZE, capthick=ERRORBAR_LINEWIDTH,
            linestyle="None", alpha=0.9, zorder=3,
        )
        handles.append(marker_handle(style, style["short"]))

    ax.set_xlabel(T_LABEL, fontsize=LABEL_FONTSIZE, labelpad=LABELPAD)
    ax.set_ylabel(cfg["dev_label"], fontsize=LABEL_FONTSIZE, labelpad=LABELPAD)
    return handles


def draw_coverage(ax, summary: dict, cfg: dict) -> list:
    """Panel (c): empirical against nominal coverage on the test split."""
    table = pd.DataFrame(summary["coverage_test"])
    nominal = table["nominal"] * 100
    empirical = table["empirical"] * 100

    ax.plot([40, 100], [40, 100], **DIAGONAL_STYLE, zorder=2)

    style = SPLIT_STYLE["test"]
    ax.plot(nominal, empirical, color=style["edge"], linewidth=1.2, zorder=3)
    ax.scatter(nominal, empirical, marker=style["marker"], s=MARKER_SIZE * 1.6,
               facecolors=style["face"], edgecolors=style["edge"],
               linewidths=MARKER_LINEWIDTH, zorder=4)

    ax.set_xlim(45, 102)
    ax.set_ylim(45, 102)
    ax.set_xlabel(NOMINAL_LABEL, fontsize=LABEL_FONTSIZE, labelpad=LABELPAD)
    ax.set_ylabel(EMPIRICAL_LABEL, fontsize=LABEL_FONTSIZE, labelpad=LABELPAD)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))

    return [
        Line2D([0], [0], **DIAGONAL_STYLE, label=bold(r"$\mathrm{Ideal}$")),
        marker_handle(style, rf"${cfg['series']}$ (test)"),
    ]


# --------------------------------------------------------------------------- #
# Assembly                                                                     #
# --------------------------------------------------------------------------- #

def build_panel(letter: str, feed: int, data_dir: str, dpi: int,
                outdirs: dict[bool, str], cfg: dict,
                crop_each: bool = False) -> None:
    frame, summary = load_intervals(data_dir, cfg)
    fig, ax = ps.plot_init()

    if letter in ("a", "b"):
        rows = feed_rows(frame, summary, feed)
        if rows.empty:
            raise SystemExit(f"panel ({letter}): no held-out points for feed {feed}")
        if letter == "a":
            # Panel (a) may be drawn on one pressure slice; panel (b) keeps
            # every held-out point, since a deviation needs no curve.
            whole = feed_frame(int(rows.source_id.iloc[0]), cfg)
            rows = slice_of(rows, whole, cfg)
            handles = draw_curve(ax, rows, reference_curve(whole, cfg), cfg)
        else:
            handles = draw_deviation(ax, rows, cfg)
        # Both quantities fall steeply with T and then flatten, and the
        # intervals of (b) are widest at the low-T end, so the free corner is
        # the top right in either case.
        loc = cfg["legend_loc"][letter]
        note = (f"feed {feed} (source_id {int(rows.source_id.iloc[0])}), "
                f"{len(rows)} held-out points")
    else:
        handles = draw_coverage(ax, summary, cfg)
        loc = "lower right"
        note = f"{summary['coverage_test'][0]['n']} test points"

    style_axes(ax)
    legend = ax.legend(handles=handles, loc=loc,
                       prop={"size": RMSE_FONTSIZE, "weight": "bold"},
                       frameon=False, borderpad=0.25,
                       handlelength=1.4, handletextpad=0.4, labelspacing=0.35)
    legend.set_zorder(6)

    fig.tight_layout()
    save_variants(fig, ax, outdirs, f"{cfg['stem']}{letter}", dpi, crop_each)
    plt.close(fig)
    print(f"  ({letter}) {note}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--panels", default="all",
                        help="comma-separated panel letters, or 'all'")
    parser.add_argument("--model", default="tabpfn", choices=sorted(MODELS),
                        help="tabpfn -> P_bubble intervals; svgp -> the IFT "
                             "intervals of the WSD + SVGP residual model")
    parser.add_argument("--feed", type=int, default=DEFAULT_FEED,
                        help=f"Table 1 feed for panels (a,b) "
                             f"(default: {DEFAULT_FEED})")
    parser.add_argument("--out", default=OUTPUT_DIR,
                        help="where the panels are written")
    parser.add_argument("--data", default=OUTPUT_DIR,
                        help="where the interval script wrote its CSV and JSON")
    parser.add_argument("--dpi", type=int, default=ps.resolution_value)
    parser.add_argument("--legend", choices=("both", "on", "off"), default="both")
    parser.add_argument("--crop-each", action="store_true",
                        help="crop each variant to its own content")
    parser.add_argument("--overleaf", action="store_true",
                        help=f"copy the legend version into {OVERLEAF_DIR}")
    args = parser.parse_args(argv)

    letters = (list(PANELS) if args.panels == "all"
               else [s.strip() for s in args.panels.split(",") if s.strip()])
    unknown = [s for s in letters if s not in PANELS]
    if unknown:
        parser.error(f"unknown panel(s): {', '.join(unknown)}")

    outdirs = {}
    if args.legend in ("both", "on"):
        outdirs[True] = os.path.join(args.out, LEGEND_SUBDIR[True])
    if args.legend in ("both", "off"):
        outdirs[False] = os.path.join(args.out, LEGEND_SUBDIR[False])

    cfg = MODELS[args.model]
    print(f"{cfg['stem']} ({args.model}): rebuilding {len(letters)} panel(s) "
          f"at {args.dpi} dpi")
    for letter in letters:
        build_panel(letter, args.feed, args.data, args.dpi, outdirs, cfg,
                    args.crop_each)

    if args.overleaf:
        src = outdirs.get(True)
        if src is None:
            parser.error("--overleaf needs the legend variant (--legend both|on)")
        for letter in letters:
            # The SI includes the vector version only.
            shutil.copy2(
                os.path.join(src, f"{cfg['stem']}{letter}.pdf"),
                os.path.join(OVERLEAF_DIR, f"{cfg['overleaf_stem']}{letter}.pdf"),
            )
        print(f"Copied {len(letters)} panel(s) into {OVERLEAF_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
