#!/usr/bin/env python
"""
tabpfn_prediction_intervals.py
==============================
Predictive quantiles of the HPO-tuned TabPFN models, for P_bubble and gamma.

Reviewer request (R2): "It would be helpful to include one example showing
95 % prediction intervals from TabPFN to illustrate model confidence."

TabPFN is a distributional regressor: the head emits a discretised density over
the target (a Riemann/bar distribution over the binned target axis), and the
point prediction reported everywhere else in the manuscript is the mean of that
density.  A prediction interval therefore needs no resampling, no ensembling and
no refit -- the 2.5 % and 97.5 % quantiles of the same density are one more
readout of the forward pass that already produced the mean
(``output_type="main"`` returns mean, median, mode and quantiles together, so
the quantiles cost nothing beyond the mean).

This script writes two files per target that ``build_figure11.py`` turns into
panels (``<TARGET>`` is ``P_bubble`` or ``gamma``):

  FIGURE11/TabPFN_<TARGET>_intervals.csv   one row per evaluated state point:
                                           split, feed, T, P, reference value,
                                           predictive mean/median and the
                                           quantile ladder
  FIGURE11/TabPFN_<TARGET>_coverage.json   empirical coverage of the central
                                           intervals on the test set, interval
                                           widths, and the reproduction check
                                           described below

Widths and errors in the JSON are in the unit of the target it belongs to,
recorded under ``"unit"``: bar for P_bubble, mN/m for gamma.

Which rows are evaluated
------------------------
The whole test split (2904 points), which is what the coverage numbers are
computed on, plus the held-out (test and validation) points of the feeds listed
in Table 1 of the manuscript, so the example panel can be drawn for a feed the
reader has already met.  Training points are deliberately excluded: an interval
drawn around a point the model was fitted on would illustrate nothing.

Those example-feed points are evaluated first and the CSV is rewritten after
every chunk, so the example panel can be drawn long before the coverage run
finishes.  The order of the rows in the CSV carries no meaning beyond that.

Provenance of the models
------------------------
The archived runs are TabPFN-snellius-home/with_HPO/SLURM{Bubble,Gamma} (100
hyperopt trials each; seed 454015 for P_bubble, whose five-fold CV RMSE of
1.31 bar is the number quoted in the Abstract, and seed 844015 for gamma, whose
CV RMSE is 0.0395 mN/m).  The seeds differ, so each target has its own split and
the two runs are not row-comparable.  The ``TabPFN_<TARGET>_model.joblib`` is
loaded for its hyperparameters, but the fitted state inside it cannot be used
directly here:
the file was pickled by the Snellius build (Python 3.13 / scikit-learn 1.7.2)
and its preprocessing steps come back missing ``subsampled_features_``, which
this build's ``ReshapeFeatureDistributionsStep.transform`` requires.  The model
is therefore rebuilt locally from those hyperparameters and re-fitted on the
same training split -- which for TabPFN is not a retraining at all: fit only
preprocesses and stores the training set as in-context data, the network
weights are the frozen v2.5 low-skew checkpoint either way.

That leaves one real difference: the archived predictions were produced on an
H100, these on CPU, and the two do not agree bit for bit (autocast dtype and
kernel order differ).  The check quantifies it against the stored
``TabPFN_<TARGET>_predictions.csv`` and the result is written to the JSON, so
the size of that gap is on record next to the intervals it belongs to.  All
numbers reported from this script -- coverage included -- are computed from the
CPU predictions in the CSV, so they are internally consistent.

Cost
----
Inference is dominated by the 13552-point in-context training set, not by the
number of query points, so the run is done in as few calls as memory allows;
``--chunk`` only bounds peak memory.  Expect tens of minutes on 24 CPU threads.

Usage
-----
    python tabpfn_prediction_intervals.py                    # P_bubble -> FIGURE11/
    python tabpfn_prediction_intervals.py --target gamma     # the IFT model
    python tabpfn_prediction_intervals.py --limit 200        # smoke test
    python tabpfn_prediction_intervals.py --no-check         # skip the H100 comparison
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# The archived model pickles CUDA storages; this machine has no GPU.
torch.load = functools.partial(torch.load, map_location="cpu")

from tabpfn import TabPFNRegressor

# --------------------------------------------------------------------------- #
# Paths and run definition                                                     #
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "interfacial_results_dataset_A4.csv")
ARCHIVE = os.path.join(HERE, "TabPFN-snellius-home", "with_HPO")
OUTPUT_DIR = os.path.join(HERE, "FIGURE11")
CHECKPOINT = os.path.expanduser(
    "~/.cache/tabpfn/tabpfn-v2.5-regressor-v2.5_low-skew.ckpt"
)

# One entry per archived run.  The seeds are those hardcoded in the run
# notebooks, so each target reproduces its own split -- they are not the same
# split, and a row index means nothing across targets.
RUNS = {
    "P_bubble": {"dir": "SLURMBubble", "seed": 454015, "unit": "bar"},
    "gamma": {"dir": "SLURMGamma", "seed": 844015, "unit": "mN/m"},
}

N_THREADS = 24

# Central intervals to score.  The ladder exists so the coverage panel can show
# whether the miscalibration (if any) is specific to the 95 % level or holds
# across the whole predictive distribution.
CENTRAL_LEVELS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
QUANTILES = sorted({
    round(q, 4)
    for lvl in CENTRAL_LEVELS
    for q in ((1.0 - lvl) / 2.0, (1.0 + lvl) / 2.0)
})

# Feeds of Table 1 of the manuscript, as mole fractions in the column order
# below.  Feed 4 is the one used for Fig. 8a.  They are matched back to the
# dataset by composition rather than by index because the table was typeset
# from the pool, not from the dataset.
TABLE1_COLUMNS = [
    "z_carbon dioxide", "z_hydrogen", "z_argon", "z_nitrogen",
    "z_methane", "z_oxygen", "z_carbon monoxide", "z_hydrogen sulfide",
]
TABLE1_FEEDS = {
    1: [0.97, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0],
    2: [0.95, 0.0313, 0.0, 0.0, 0.0039, 0.0, 0.0, 0.0148],
    3: [0.98, 0.0058, 0.0091, 0.0011, 0.0029, 0.0011, 0.0, 0.0],
    4: [0.96, 0.0200, 0.0, 0.0080, 0.0040, 0.0040, 0.0020, 0.0020],
    5: [0.97, 0.0075, 0.0040, 0.0079, 0.0099, 3.79e-5, 6.99e-4, 3.90e-7],
}
FEED_MATCH_TOL = 1e-4   # max |dz| accepted when matching a table feed to a feed
                        # of the dataset; feed 4 does not match at this tolerance


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #

def run_config(target: str) -> dict:
    """Paths, seed and unit of the archived run for ``target``."""
    run = dict(RUNS[target], target=target)
    run["path"] = os.path.join(ARCHIVE, run["dir"])
    run["model"] = os.path.join(run["path"], f"TabPFN_{target}_model.joblib")
    run["stored_pred"] = os.path.join(run["path"],
                                      f"TabPFN_{target}_predictions.csv")
    return run


def load_split(run: dict) -> tuple:
    """Rebuild the exact 70/15/15 split of the archived run."""
    df = pd.read_csv(DATASET)
    z_nonzero = [c for c in df.columns
                 if c.startswith("z_") and (df[c] != 0).any()]
    features = ["temperature", "pressure"] + z_nonzero

    X, y = df[features], df[run["target"]]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=run["seed"])
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=run["seed"])
    return df, features, X_train, y_train, X_test, y_test, X_val, y_val


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
# Model                                                                        #
# --------------------------------------------------------------------------- #

def rebuild_model(run: dict) -> TabPFNRegressor:
    """The archived model's hyperparameters, instantiated for this machine."""
    params = joblib.load(run["model"]).get_params()
    params["device"] = "cpu"
    params["model_path"] = CHECKPOINT      # the archived path is Snellius'
    return TabPFNRegressor(**params)


def predict_intervals(model: TabPFNRegressor, X: pd.DataFrame, chunk: int,
                      on_chunk=None) -> pd.DataFrame:
    """Mean, median and the quantile ladder for every row of ``X``.

    Inference cost grows with the number of query points, so this is the long
    part of the run; ``on_chunk`` is called with the rows completed so far, to
    checkpoint them.
    """
    done = []
    for start in range(0, len(X), chunk):
        part = X.iloc[start:start + chunk]
        t0 = time.perf_counter()
        out = model.predict(part, output_type="main", quantiles=QUANTILES)
        frame = pd.DataFrame(
            {"mean": np.asarray(out["mean"]), "median": np.asarray(out["median"])},
            index=part.index,
        )
        for j, q in enumerate(QUANTILES):
            frame[f"q{q:.4f}"] = np.asarray(out["quantiles"][j])
        done.append(frame)
        print(f"  rows {start}-{start + len(part)} of {len(X)}: "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        if on_chunk is not None:
            on_chunk(pd.concat(done))
    return pd.concat(done)


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #

def coverage_table(frame: pd.DataFrame) -> list[dict]:
    """Empirical coverage of each central interval, on the given rows."""
    rows = []
    for lvl in CENTRAL_LEVELS:
        lo = frame[f"q{(1.0 - lvl) / 2.0:.4f}"].to_numpy()
        hi = frame[f"q{(1.0 + lvl) / 2.0:.4f}"].to_numpy()
        y = frame["actual"].to_numpy()
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


def reproduction_check(frame: pd.DataFrame, run: dict) -> dict | None:
    """How far these CPU means sit from the archived H100 predictions."""
    if not os.path.exists(run["stored_pred"]):
        return None
    stored = pd.read_csv(run["stored_pred"]).set_index("idx")
    common = frame.index.intersection(stored.index)
    if not len(common):
        return None
    mine = frame.loc[common, "mean"].to_numpy()
    theirs = stored.loc[common, "predicted"].to_numpy()
    actual = stored.loc[common, "actual"].to_numpy()
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    return {
        "n_compared": int(len(common)),
        "max_abs_diff": float(np.abs(mine - theirs).max()),
        "rmse_between_runs": rmse(mine, theirs),
        "rmse_vs_reference_cpu": rmse(mine, actual),
        "rmse_vs_reference_h100": rmse(theirs, actual),
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="P_bubble", choices=sorted(RUNS),
                        help="archived run to read the intervals from")
    parser.add_argument("--out", default=OUTPUT_DIR,
                        help="output directory (default: FIGURE11/)")
    parser.add_argument("--chunk", type=int, default=1500,
                        help="query points per forward pass; bounds peak memory")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N test points (smoke test)")
    parser.add_argument("--threads", type=int, default=N_THREADS)
    parser.add_argument("--no-check", action="store_true",
                        help="skip the comparison against the archived run")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    run = run_config(args.target)
    target, unit = run["target"], run["unit"]
    print(f"Target {target} ({unit}), archived run {run['dir']}, "
          f"seed {run['seed']}")

    print("Loading dataset and rebuilding the archived split ...")
    df, features, X_train, y_train, X_test, y_test, X_val, y_val = load_split(run)
    print(f"  train {len(X_train)}  test {len(X_test)}  val {len(X_val)}")

    feeds = match_table1_feeds(df)
    print(f"  Table 1 feeds present: "
          f"{', '.join(f'{k} -> source_id {v}' for k, v in feeds.items())}")

    # The held-out points of the example feeds go first so the example panel can
    # be drawn while the rest of the test split is still running; the remaining
    # test points follow.  Both splits are held out, so both may carry an
    # interval, but the coverage numbers below use the test rows only.
    on_feed = df["source_id"].isin(feeds.values())
    feed_idx = (X_test.index[on_feed.loc[X_test.index]]
                .append(X_val.index[on_feed.loc[X_val.index]]))
    test_idx = X_test.index.difference(feed_idx)
    if args.limit:
        test_idx = test_idx[:args.limit]
    idx = feed_idx.append(test_idx)

    split = pd.Series("test", index=idx)
    split.loc[idx.intersection(X_val.index)] = "val"

    print(f"Predicting {len(idx)} points "
          f"({(split == 'test').sum()} test, {(split == 'val').sum()} val; "
          f"{len(feed_idx)} of them on the example feeds, evaluated first) ...")
    model = rebuild_model(run)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"  fit (in-context, no weight update): {time.perf_counter() - t0:.1f}s")

    csv_path = os.path.join(args.out, f"TabPFN_{target}_intervals.csv")

    def assemble(pred: pd.DataFrame) -> pd.DataFrame:
        """Predictions joined to the state point they belong to."""
        rows = df.loc[pred.index]
        out = pd.DataFrame(
            {
                "split": split.loc[pred.index],
                "source_id": rows["source_id"],
                "temperature": rows["temperature"],
                "pressure": rows["pressure"],
                "actual": rows[target],
            },
        ).join(pred)
        out.index.name = "idx"
        return out

    t0 = time.perf_counter()
    pred = predict_intervals(
        model, df.loc[idx, features], args.chunk,
        on_chunk=lambda part: assemble(part).to_csv(csv_path),
    )
    print(f"  inference total: {(time.perf_counter() - t0) / 60:.1f} min")

    out = assemble(pred)
    out.to_csv(csv_path)
    print(f"Wrote {csv_path}")

    summary = {
        "model": "TabPFN with HPO",
        "run": os.path.relpath(run["path"], HERE),
        "target": target,
        "unit": unit,
        "seed": run["seed"],
        "device": "cpu",
        "n_train": int(len(X_train)),
        "features": features,
        "quantiles": QUANTILES,
        "coverage_test": coverage_table(out[out.split == "test"]),
        "table1_feed_to_source_id": feeds,
        "test_rmse": float(np.sqrt(np.mean(
            (out.loc[out.split == "test", "mean"]
             - out.loc[out.split == "test", "actual"]) ** 2))),
    }
    if not args.no_check:
        summary["reproduction_check"] = reproduction_check(out, run)

    json_path = os.path.join(args.out, f"TabPFN_{target}_coverage.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {json_path}")

    print("\nCoverage of the central intervals (test split):")
    for row in summary["coverage_test"]:
        print(f"  nominal {row['nominal']:.0%}  empirical "
              f"{row['empirical']:.1%}  median width "
              f"{row['median_width']:.4g} {unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
