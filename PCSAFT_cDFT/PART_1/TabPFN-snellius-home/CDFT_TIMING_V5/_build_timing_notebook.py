#!/usr/bin/env python3
"""Build an instrumented copy of VLE_IFT_V5.ipynb that records per-stage wall
time (phase-equilibria vs cDFT interfacial sweep) for a single feed and writes
a per-feed timing JSON + CSV. The physics is byte-for-byte the original V5; we
only wrap the two heavy cells with perf_counter markers and append a writer
cell. Run once with py_A6:  python _build_timing_notebook.py
"""
import json, uuid
from pathlib import Path

SRC = Path("../../SENSITIVITY_ANALYSIS/VLE_IFT_V5.ipynb")
DST = Path("VLE_IFT_V5_TIMED.ipynb")

nb = json.loads(SRC.read_text())


def code_cell(src):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:12],
        "metadata": {"tags": ["timing-instrumentation"]},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


# Ensure every cell has an id (nbformat 4.5 requirement) and find anchors by the
# first non-empty source line, so the build is robust to index drift.
def first_line(cell):
    for ln in cell["source"]:
        s = ln.strip()
        if s:
            return s
    return ""


for c in nb["cells"]:
    c.setdefault("id", uuid.uuid4().hex[:12])

# 1) Append TIMING_FILE parameter to the parameters-tagged cell.
for c in nb["cells"]:
    if c["cell_type"] == "code" and "parameters" in c.get("metadata", {}).get("tags", []):
        c["source"].append("\nTIMING_FILE                     = None   # output path for per-feed timing JSON (set by SLURM script)\n")
        break

PRE_PHASE = code_cell(
    "# [timing] phase-equilibria stage START\n"
    "import time as _time\n"
    "_t_phase_start = _time.perf_counter()\n"
)
POST_PHASE = code_cell(
    "# [timing] phase-equilibria stage END\n"
    "_t_phase_end = _time.perf_counter()\n"
)
PRE_CDFT = code_cell(
    "# [timing] cDFT interfacial sweep START\n"
    "_t_cdft_start = _time.perf_counter()\n"
)
POST_CDFT = code_cell(
    "# [timing] cDFT interfacial sweep END\n"
    "_t_cdft_end = _time.perf_counter()\n"
)

WRITER = code_cell(r'''# [timing] aggregate per-stage wall time for this feed and persist it.
import json as _json, os as _os, csv as _csv
from pathlib import Path as _Path

_phase_eq_s = _t_phase_end - _t_phase_start
_cdft_s     = _t_cdft_end - _t_cdft_start
_total_s    = _phase_eq_s + _cdft_s

# Count solved (T,P) points and isotherms across whatever feed(s) ran.
_n_points = 0
_n_ok     = 0
_n_iso    = 0
_feeds_meta = {}
for _fk, _fv in VLE_DFT.items():
    _idat = _fv.get("interfacial_data", {})
    _n_iso += len(_idat)
    for _T, _rows in _idat.items():
        for _r in _rows:
            _n_points += 1
            _g = _r.get("gamma_mN_m", _r.get("gamma")) if isinstance(_r, dict) else None
            if _g is not None and _g == _g:   # not NaN
                _n_ok += 1
    _pt = PT_results.get(_fk, {})
    _feeds_meta[_fk] = {
        "z": _fv.get("z"),
        "active_components": _pt.get("active_components"),
        "TC_K": _pt.get("TC_K"),
        "PC_bar": _pt.get("PC_bar"),
    }

def _per_pt_ms(sec):
    return 1000.0 * sec / _n_points if _n_points else float("nan")

_timing = {
    "feed_index": FEED_INDEX,
    "version": "V5_contour_map",
    "grid": {"ngrid": ngrid, "lgrid": lgrid, "T_STEP": T_STEP,
             "cDFT_NP": cDFT_NP, "P_TOL": P_TOL, "SLURM_RUN": SLURM_RUN},
    "critical_region_enhancements": CRITICAL_REGION_ENHANCEMENTS,
    "threads": {
        "NUM_THREADS_outer_pool": NUM_THREADS_RESOLVED,
        "RAYON_NUM_THREADS": _os.environ.get("RAYON_NUM_THREADS"),
        "OMP_NUM_THREADS": _os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": _os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": _os.environ.get("MKL_NUM_THREADS"),
        "SLURM_CPUS_PER_TASK": _os.environ.get("SLURM_CPUS_PER_TASK"),
    },
    "node": _os.environ.get("SLURMD_NODENAME") or _os.uname().nodename,
    "counts": {"n_points": _n_points, "n_points_converged": _n_ok, "n_isotherms": _n_iso},
    "timing_s": {"phase_equilibria": _phase_eq_s, "cdft_sweep": _cdft_s, "total_stage": _total_s},
    "per_point_ms": {
        "phase_equilibria_amortized": _per_pt_ms(_phase_eq_s),
        "cdft_sweep": _per_pt_ms(_cdft_s),
        "total": _per_pt_ms(_total_s),
    },
    "per_point_s": {
        "cdft_sweep": (_cdft_s / _n_points) if _n_points else float("nan"),
        "total": (_total_s / _n_points) if _n_points else float("nan"),
    },
    "feeds": _feeds_meta,
}

_out = TIMING_FILE
if _out is None:
    _out = _os.path.join(CSV_FOLDER, f"timing_feed_{FEED_INDEX}.json")
_Path(_out).parent.mkdir(parents=True, exist_ok=True)
with open(_out, "w") as _f:
    _json.dump(_timing, _f, indent=2)

# Flat one-row CSV alongside the JSON.
_csv_out = _os.path.splitext(_out)[0] + ".csv"
_row = {
    "feed_index": FEED_INDEX,
    "n_points": _n_points,
    "n_points_converged": _n_ok,
    "n_isotherms": _n_iso,
    "ngrid": ngrid, "lgrid": lgrid, "T_STEP": T_STEP, "cDFT_NP": cDFT_NP,
    "outer_workers": NUM_THREADS_RESOLVED,
    "rayon_threads": _os.environ.get("RAYON_NUM_THREADS"),
    "phase_eq_s": round(_phase_eq_s, 4),
    "cdft_sweep_s": round(_cdft_s, 4),
    "total_stage_s": round(_total_s, 4),
    "cdft_per_point_s": round(_cdft_s / _n_points, 6) if _n_points else "",
    "total_per_point_s": round(_total_s / _n_points, 6) if _n_points else "",
    "node": _timing["node"],
}
with open(_csv_out, "w", newline="") as _f:
    _w = _csv.DictWriter(_f, fieldnames=list(_row.keys()))
    _w.writeheader(); _w.writerow(_row)

print("=== TIMING SUMMARY (feed", FEED_INDEX, ") ===")
print(f"  points solved     : {_n_points}  (converged {_n_ok}, isotherms {_n_iso})")
print(f"  phase equilibria  : {_phase_eq_s:10.2f} s  ({_per_pt_ms(_phase_eq_s):.4f} ms/pt amortized)")
print(f"  cDFT sweep        : {_cdft_s:10.2f} s  ({_cdft_s/_n_points if _n_points else float('nan'):.4f} s/pt)")
print(f"  total stage       : {_total_s:10.2f} s  ({_total_s/_n_points if _n_points else float('nan'):.4f} s/pt)")
print(f"  wrote {_out}")
print(f"  wrote {_csv_out}")
''')

# Rebuild cell list, inserting timers around the phase-equilibria cell (anchor
# "PT_results = {}") and the cDFT cell (anchor starts with "# Parallelized").
out_cells = []
for c in nb["cells"]:
    fl = first_line(c)
    if c["cell_type"] == "code" and fl.startswith("PT_results = {}"):
        out_cells.append(PRE_PHASE)
        out_cells.append(c)
        out_cells.append(POST_PHASE)
    elif c["cell_type"] == "code" and fl.startswith("# Parallelized (T, P) cDFT sweep"):
        out_cells.append(PRE_CDFT)
        out_cells.append(c)
        out_cells.append(POST_CDFT)
    else:
        out_cells.append(c)

out_cells.append(WRITER)
nb["cells"] = out_cells

DST.write_text(json.dumps(nb, indent=1))
print(f"wrote {DST} with {len(nb['cells'])} cells")
