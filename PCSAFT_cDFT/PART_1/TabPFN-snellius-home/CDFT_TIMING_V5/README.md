# CDFT_TIMING_V5 — re-time the cDFT γ-map, phase-eq vs cDFT, per feed

Purpose: re-run the **V5 contour-map** sensitivity sweep for the **same 5 study
feeds** and record where the per-state-point seconds actually go — splitting the
**phase-equilibria stage** (critical point + bubble/dew envelope, once per feed)
from the **cDFT interfacial sweep** (TP-flash + planar-interface solve at every
(T,P) point). This is what makes our 0.75–6.57 s/point cDFT cost comparable to
literature numbers (e.g. Rehner 2023's ~0.09 s 1D surface tension), which exclude
the multicomponent flash and use a single simple interface.

## What's here
- `VLE_IFT_V5_TIMED.ipynb` — byte-for-byte the physics of
  `../../SENSITIVITY_ANALYSIS/VLE_IFT_V5.ipynb`, with timing markers wrapping the
  phase-equilibria cell and the cDFT-sweep cell, plus a writer cell that dumps a
  per-feed timing JSON/CSV. (Regenerate with `python _build_timing_notebook.py`.)
- `RUN_TIMING_V5.sh` — SLURM array job for c109.
- `aggregate_timing.py` — combine the per-feed JSONs into `cdft_stage_timing.{csv,md}`.
- `CSV_feeds/` — the 5 feed compositions + KIJ pairs (copied from SENSITIVITY_ANALYSIS).

## Configuration (as built)
- **Default grid** (`SLURM_RUN=False`): ngrid=500, lgrid=100 Å, T_STEP=5 K, cDFT_NP=5.
  This is the modest default, not the production grid (ngrid=2048, T_STEP=0.1,
  cDFT_NP=10) that produced the original multi-hour maps — chosen for a quick,
  clean per-stage breakdown.
- **No critical-region enhancement** (`CRITICAL_REGION_ENHANCEMENTS=False`).
- **Hybrid threading**: OUTER Python ThreadPool workers × INNER rayon/BLAS threads,
  with OUTER×INNER = `cpus-per-task` (default 4×4 on 16 CPUs). Note the original
  `RUN.sh` used outer-only (rayon=1); this run enables rayon per the request, so
  per-point wall times are **not** directly comparable to that config — the
  phase-eq-vs-cDFT *split* is the robust takeaway.

## Submit (5 feeds as array 0-4, on c109)
```bash
cd /home/darshan/A6/PCSAFT_cDFT/PART_1/TabPFN-snellius-home/CDFT_TIMING_V5
sbatch RUN_TIMING_V5.sh                  # default hybrid 4x4
# OUTER=2 INNER=8 sbatch RUN_TIMING_V5.sh # override the split
```
The first invocation resubmits itself as an array `0-4`. Per-feed timing lands in
`RESULTS/timing/timing_feed_<i>.json`; full notebooks + CSVs under `RESULTS/<jobid>_<i>/`.

## After it finishes
```bash
python aggregate_timing.py   # writes cdft_stage_timing.csv and .md
```

## Smoke test (already done locally, feed 0, default grid)
95 points / 19 isotherms: phase-eq 2.9 s, cDFT sweep 24.6 s (total 27.6 s) →
phase equilibria ≈ 11% of stage time, cDFT ≈ 89%; ~0.29 s/point total at ngrid=500.
(Local hal9000 4×4; c109 numbers will differ — that's the point of the job.)
