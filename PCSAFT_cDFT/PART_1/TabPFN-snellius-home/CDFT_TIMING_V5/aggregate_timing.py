#!/usr/bin/env python3
"""Combine the per-feed timing JSONs written by the array job into one CSV + MD.
Run after the SLURM array finishes:  python aggregate_timing.py
"""
import json, csv, glob, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
TIMING_DIR = os.path.join(HERE, "RESULTS", "timing")
OUT_CSV = os.path.join(HERE, "cdft_stage_timing.csv")
OUT_MD = os.path.join(HERE, "cdft_stage_timing.md")

files = sorted(glob.glob(os.path.join(TIMING_DIR, "timing_feed_*.json")))
if not files:
    raise SystemExit(f"No timing JSONs found in {TIMING_DIR} -- has the job run?")

rows = []
for f in files:
    d = json.load(open(f))
    c = d["counts"]; t = d["timing_s"]; pp = d["per_point_s"]
    rows.append({
        "feed": d["feed_index"],
        "n_points": c["n_points"],
        "n_isotherms": c["n_isotherms"],
        "phase_eq_s": t["phase_equilibria"],
        "cdft_sweep_s": t["cdft_sweep"],
        "total_stage_s": t["total_stage"],
        "phase_eq_per_point_ms": d["per_point_ms"]["phase_equilibria_amortized"],
        "cdft_per_point_s": pp["cdft_sweep"],
        "total_per_point_s": pp["total"],
        "outer_workers": d["threads"]["NUM_THREADS_outer_pool"],
        "rayon_threads": d["threads"]["RAYON_NUM_THREADS"],
        "node": d.get("node"),
    })

with open(OUT_CSV, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

def fmt(x, n=4):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) and not math.isnan(x) else str(x)

tot_pts = sum(r["n_points"] for r in rows)
tot_phase = sum(r["phase_eq_s"] for r in rows)
tot_cdft = sum(r["cdft_sweep_s"] for r in rows)

with open(OUT_MD, "w") as fh:
    fh.write("# cDFT map timing — phase equilibria vs cDFT sweep (per feed)\n\n")
    fh.write("Same physics as `VLE_IFT_V5.ipynb`, default grid (ngrid=500, T_STEP=5, "
             "cDFT_NP=5), no critical-region enhancement, hybrid threading on c109.\n")
    fh.write("Phase-equilibria time is computed **once per feed** (critical point + bubble/dew "
             "envelope); the per-point column amortizes it over the map points. The cDFT-sweep "
             "time is the parallel TP-flash + planar-interface solve over all (T,P) points.\n\n")
    fh.write("| feed | points | isotherms | phase-eq (s) | cDFT sweep (s) | total (s) | "
             "phase-eq/pt (ms, amort.) | cDFT/pt (s) | total/pt (s) |\n")
    fh.write("|---|--:|--:|--:|--:|--:|--:|--:|--:|\n")
    for r in rows:
        fh.write(f"| {r['feed']} | {r['n_points']} | {r['n_isotherms']} | "
                 f"{fmt(r['phase_eq_s'],1)} | {fmt(r['cdft_sweep_s'],1)} | {fmt(r['total_stage_s'],1)} | "
                 f"{fmt(r['phase_eq_per_point_ms'],4)} | {fmt(r['cdft_per_point_s'],4)} | "
                 f"{fmt(r['total_per_point_s'],4)} |\n")
    fh.write(f"\n**Totals:** {tot_pts} points, phase-eq {tot_phase:.1f} s, "
             f"cDFT sweep {tot_cdft:.1f} s. "
             f"Phase-eq is {100*tot_phase/(tot_phase+tot_cdft):.1f}% of the combined stage time.\n")
    fh.write(f"\nNode: {rows[0].get('node')} | outer workers {rows[0]['outer_workers']} "
             f"x rayon {rows[0]['rayon_threads']}.\n")

print(f"wrote {OUT_CSV}")
print(f"wrote {OUT_MD}")
print(f"\n{len(rows)} feeds | {tot_pts} points | phase-eq {tot_phase:.1f}s | cDFT {tot_cdft:.1f}s")
