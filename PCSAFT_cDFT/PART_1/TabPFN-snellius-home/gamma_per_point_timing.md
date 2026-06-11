# γ — inference time per state point (cDFT vs all surrogates)

Per-state-point cost of computing the surface tension γ: the physics engine (cDFT +
PC-SAFT) versus every ML surrogate, **for γ alone**. Data in `gamma_per_point_timing.csv`.

## Table

| model | family | mode | time / point | speed-up vs cDFT | hardware |
|---|---|---|--:|--:|---|
| **cDFT (PC-SAFT)** | physics | per point (min–max) | 0.75 – 6.57 s (mean 3.48 s) | 1× | Snellius CPU/BLAS |
| TabPFN without_HPO | surrogate | isolated | 0.73 ms | ~4 700× | H100 GPU |
| TabPFN with_HPO | surrogate | isolated | 1.74 ms | ~2 000× | H100 GPU |
| TabPFN with_HPO | surrogate | batched (deployed) | 0.48 ms | ~7 200× | H100 GPU |
| RF + TabPFN | surrogate | isolated | 260.2 ms | ~13× | H100 GPU |
| **GPR** (RBF+White) | residual | isolated | 0.86 ms | ~4 000× | py_A6 CPU |
| **GPR** (RBF+White) | residual | batched | 0.0086 ms | ~406 000× | py_A6 CPU |
| **SVGP** (Matérn-5/2 ARD) | residual | isolated | 4.99 ms | ~700× | py_A6 CPU |
| **SVGP** (Matérn-5/2 ARD) | residual | batched | 0.047 ms | ~74 000× | py_A6 CPU |
| **SR** (closed-form eq) | residual | batched | ~8×10⁻⁶ ms | ~4×10⁸× | py_A6 CPU |

## How to read it

- **cDFT** is the physics baseline: **seconds per single state point** (0.75 s on the
  cheapest feed, 6.57 s on the most expensive — a ~9× spread), mean ≈ 3.48 s/point.
- **TabPFN surrogates** (the γ-map study): plain variants turn that into well under
  1 ms/point (10³–10⁴×). `with_HPO` is the deployed model; its **batched** 0.48 ms/point
  is the number that produced the 3 000–17 000× speed-ups in the cDFT-vs-ML study.
  `RF+TabPFN` is the outlier at 260 ms/point (~13×) — TabPFN is re-run at every RF leaf.
- **Residual models** (GPR / SVGP / SR) predict the *cDFT − analytic* residual, so γ also
  needs the cheap analytic γ_wsd term; the numbers below are the **residual model alone**.
  - **GPR** is a single linear-algebra solve at inference: batched it is ~0.0086 ms/point
    (the cheapest learned model here), ~0.86 ms/point isolated.
  - **SVGP** (2000 inducing points, Matérn-5/2 ARD) is ~0.047 ms/point batched, ~5 ms/point
    isolated — the per-call overhead of a 2000-point variational forward dominates the
    isolated number; batched it is still well under 0.1 ms/point.
  - **SR** is a closed-form algebraic equation — essentially free (~8 ns/point); the
    speed-up figure is illustrative, not a meaningful engineering limit.

## ⚠️ Caveats (important before quoting these)

1. **Mixed hardware.** cDFT and TabPFN were timed on **Snellius (H100 node)**; the three
   residual models (GPR / SVGP / SR) were all timed together in the **`py_A6` env on the
   local CPU — Intel Xeon Silver 4214R @ 2.40 GHz (2 sockets × 12 cores = 24 threads)**,
   torch 2.10 CPU. The residual numbers are internally consistent
   with each other but are *order-of-magnitude* against the H100 rows, not same-rig. Re-run
   the residual models on the H100 node for a strict apples-to-apples table.
2. **Isolated vs batched.** "Isolated" = one-sample `.predict()` call (includes Python/call
   overhead); "batched" = amortized per-point cost when predicting the whole map at once.
   Batched is the realistic deployment cost; isolated is the worst case.
3. **GPR batched** excludes predictive-variance (`return_std`) cost — mean prediction only.
4. **SR** time is closed-form arithmetic on the engineered features; it excludes feature
   construction (negligible) and is dominated by interpreter overhead at small batches.

## Sources
- cDFT + TabPFN: `cdft_vs_ml_bundle/cdft_vs_ml_timing.csv`, `BENCHMARK_TIMING/benchmark_timing.csv`.
- Residual models: `../RESIDUAL/GPR/GPR_RESIDUAL_OUTPUTS/GPR_residual_model.joblib`,
  `../RESIDUAL/SVGP/SVGP_RESIDUAL_OUTPUTS/SVGP_residual_model.pt`,
  `../RESIDUAL/SR/.../SR_residual_metrics.json` (equation). GPR/SR timed on this machine (CPU).
