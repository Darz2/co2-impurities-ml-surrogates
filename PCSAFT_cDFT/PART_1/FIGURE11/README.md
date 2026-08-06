# Prediction intervals (SI Figures S26 and S27)

Notes on the figures produced for **Reviewer 2, comment 4** of the A6 rebuttal:

> *"It would be helpful to include one example showing 95 % prediction intervals
> from TabPFN to illustrate model confidence."*

Both figures live in the **Supporting Information, Section S4.3** (`si.tex`), as
`FigureS26a-c.pdf` (TabPFN, P<sub>bubble</sub>) and `FigureS27a-c.pdf`
(WSD + SVGP, γ). The main manuscript carries only two sentences in Section 3.4
and a two-sentence pointer in Section 3.6, because Reviewer 1 asked for a
shorter paper.

---

## Why two different models

This is the single most important thing to remember about these figures.

**P<sub>bubble</sub> — TabPFN, intervals for free.** TabPFN is a *distributional*
regressor: its head emits a discretised (bar / Riemann) density over the target,
and the point prediction reported everywhere else in the manuscript is the
**mean** of that density. Quantiles are another readout of the same forward
pass — no retraining, no resampling, no calibration, no ensembling.

```python
out = model.predict(X, output_type="main", quantiles=[0.025, 0.975])
out["mean"]        # what the manuscript reports as the prediction
out["quantiles"]   # [q_2.5%, q_97.5%] -> the central 95 % interval
```

**γ — TabPFN cannot do this.** The archived γ model is *not* a plain TabPFN. Of
the four headline `with_HPO` runs:

| run | class | quantiles? |
|---|---|---|
| `SLURMBubble` (P<sub>bubble</sub>) | `TabPFNRegressor` | ✅ free |
| `SLURMTHICKNESS` (L₁₀⁹⁰) | `TabPFNRegressor` | ✅ free |
| `SLURMGamma` (γ) | `DecisionTreeTabPFNRegressor` | ❌ |
| `SLURMDew` (P<sub>dew</sub>) | `DecisionTreeTabPFNRegressor` | ❌ |

γ and P<sub>dew</sub> use the `dt_pfn` architecture from `tabpfn_extensions`,
matching the SI hyperparameter table (dt_pfn + `small-samples` checkpoint for γ).
That wrapper:

1. fits a CART tree over the features (γ: `max_depth=2`, `min_samples_split=1000`,
   so with 13552 training rows it genuinely splits into several nodes);
2. fits a **separate TabPFN at each node**, on that node's training subset;
3. combines the node outputs in *logit* space (`average_logits=True` for γ) while
   `adaptive_tree=True` prunes or overrides nodes scored on a held-out slice.

`predict()` is typed `-> np.ndarray` and returns point values only. There is no
`output_type` argument and no quantile path, because what comes back is no longer
one TabPFN density but a pruned, logit-averaged aggregate of several.
Reconstructing a mixture quantile from it would be our own construction, not the
model's — so the SI says this plainly rather than substituting a different model
silently.

**γ — SVGP instead.** The paper already has a genuinely probabilistic γ model:
the WSD + SVGP residual correction of Section 3.6, whose predictive standard
deviation is what Figure 10d plots. Intervals come from its posterior predictive.

---

## Where the intervals come from

### TabPFN, P<sub>bubble</sub>

| item | value |
|---|---|
| run | `TabPFN-snellius-home/with_HPO/SLURMBubble` (CV RMSE 1.3131 bar = the Abstract's "1.31 bar") |
| checkpoint | TabPFN v2.5, `v2.5_low-skew` |
| seed / split | 454015 → 13552 train / 2904 test / 2905 val |
| interval | `[q_(1−c)/2, q_(1+c)/2]` straight from the predictive density |

### SVGP, γ

| item | value |
|---|---|
| run | `RESIDUAL/SVGP/SVGP_RESIDUAL_OUTPUTS/SVGP_residual_model.pt` (Matérn-5/2 ARD, 2000 inducing points) |
| dataset | `RESIDUAL/CombinedDatasetSEC_A4.csv` — **different** from the data-driven set |
| seed / split | 4555525 → 16130 train / 3456 test / 3457 val |
| target | residual Δγ = γ<sub>cDFT</sub> − γ<sub>WSD</sub>, in a normalised, quantile-transformed space |

The SVGP interval is formed where the model is Gaussian and mapped back:

> `[μ − zσ, μ + zσ]` with `z = Φ⁻¹((1+c)/2)`, endpoints pushed through the
> **inverse quantile transform**.

The transform is monotone, so mapping the endpoints maps the quantiles *exactly*
— unlike propagating σ through its Jacobian (what `export_svgp_predictions.py`
does for the Figure 10 surfaces), which linearises and forces symmetry. The
resulting intervals in mN/m are asymmetric, as TabPFN's are. `likelihood(model(x))`
is used rather than `model(x)`, so the observation-noise term is included — these
are *predictive* intervals, not latent-posterior ones. γ<sub>WSD</sub> is
deterministic, so the interval on the reconstructed IFT is the residual interval
shifted by the baseline.

**Seeds differ between every model here** (454015 / 844015 / 4555525). The splits
are not the same, and a row index means nothing across targets.

---

## The three panels

### (a) Reference curve with the interval

Feed 2 of the compositions table (Table 2 of the revised manuscript), the feed
with the largest H₂ content (*z*<sub>H2</sub> = 0.0313).

- **P<sub>bubble</sub>**: P<sub>bubble</sub>(T) is a function of (T, z) alone, so
  all state points of the feed sit on one curve and a held-out point exists at
  nearly every temperature → shaded band (the per-temperature envelope).
- **γ**: γ depends on P as well, *strongly* at low T — at 208 K it runs from
  20.2 mN/m at 4 bar to 15.8 mN/m at 134 bar. A γ-vs-T band would fold that
  physical spread into what should read as model uncertainty. Each feed is
  sampled on a clean 5-point fractional pressure grid (0, ¼, ½, ¾, 1 between the
  dew and bubble pressures) at every temperature, so panel (a) takes the last of
  those — **γ along the bubble curve**, the same slice Figure S26 is drawn on.
  Only 10 held-out points survive that cut, too few to interpolate a band, so the
  interval is drawn per point.

### (b) The same points as deviations

Prediction minus reference with the 95 % interval as error bars — asymmetric,
since neither predictive density is Gaussian in the target space. This panel
exists because a 0.34 bar interval is invisible on an 80–163 bar axis. For γ it
keeps **all 65** held-out points of the feed, since a deviation needs no curve.

Symbols are deliberately distinct: **ΔP<sub>bubble</sub>** = prediction −
reference, but for the IFT **Δγ** already means the cDFT − WSD residual in
Figure 10, so the deviation is **δγ**.

### (c) Coverage — the calibration check

For each nominal level *c*, build the central interval at every test point and
count how often the cDFT reference lands inside:

> empirical coverage = (test points where reference ∈ interval) / n

The dashed line is *y* = *x*. **Above** = conservative (intervals wider than
needed). **Below** = overconfident, the dangerous direction.

| nominal | TabPFN P<sub>bubble</sub> (n=2904) | median width | SVGP γ (n=3456) | median width |
|--:|--:|--:|--:|--:|
| 50 % | 97.66 % | 0.114 bar | 78.1 % | 0.0298 mN/m |
| 60 % | 99.00 % | 0.143 bar | 82.6 % | 0.0370 mN/m |
| 70 % | 99.69 % | 0.176 bar | 85.9 % | 0.0456 mN/m |
| 80 % | 99.90 % | 0.218 bar | 88.7 % | 0.0564 mN/m |
| 90 % | 100.0 % | 0.282 bar | 91.5 % | 0.0727 mN/m |
| 95 % | 100.0 % | 0.339 bar | 93.1 % | 0.0866 mN/m |
| 99 % | 100.0 % | 0.455 bar | 95.5 % | 0.1139 mN/m |

**The two models behave differently, and that is the interesting result.**

- **TabPFN is conservative at every level.** Zero of 2904 test points fall
  outside the nominal 95 % band; even the interquartile band captures 97.7 %. So
  its 95 % interval is *not* a one-in-twenty statement here — the width overstates
  the true error. Likely causes: the targets are noiseless cDFT computations
  while TabPFN's synthetic pre-training prior budgets variance for observation
  noise that isn't there, and the test points come from the same 100 feeds as
  training (interpolation, not a stress test).
- **The SVGP curve crosses the diagonal**, between nominal 90 % and 95 %:
  conservative at the low levels, slightly *overconfident* at the high ones
  (93.1 % at nominal 95 %, 95.5 % at nominal 99 %). Its predictive distribution
  is Gaussian in the transformed space and so has lighter tails than the observed
  residual distribution, whose worst outliers sit at the low-T, high-P states.

Practical upshot, and what the SI says: use interval **width** as a *relative*
indicator of where a model is least certain; only the SVGP's nominal 95 %
interval is anywhere near a probability statement, and it is marginally below it.

---

## Feed-2 numbers quoted in the text

| | TabPFN, P<sub>bubble</sub> | SVGP, γ |
|---|---|---|
| held-out points on the feed | 61 | 65 (10 on the panel-(a) slice) |
| reference inside 95 % PI | 61 / 61 | 62 / 65 |
| largest deviation of the mean | 0.51 bar | 1.37 mN/m |
| interval at the low-T end | 20.4 bar at 200 K (P<sub>bubble</sub> = 162.9 bar) | median 0.68 mN/m below 215 K |
| interval at the high-T end | 0.57 bar at 265 K | median 0.13 mN/m above 235 K |

Both widen where Fig. 4a-d residuals are largest (low T, high H₂) and in the
sparsely sampled high-pressure region already flagged by the GPR predictive
uncertainty in Section 3.6.

**What panel (c) does not test:** extrapolation outside the training ranges of
Table 1. The manuscript flags that limitation separately.

---

## Reproducing

```bash
# TabPFN, ~23 min on 24 CPU threads
python tabpfn_prediction_intervals.py                 # --target gamma also works,
                                                      # but see the dt_pfn caveat
# SVGP, seconds
python svgp_prediction_intervals.py

# plotting only; --overleaf copies the PDFs into the SI
python build_figure11.py --overleaf                   # FigureS26a-c
python build_figure11.py --model svgp --overleaf      # FigureS27a-c
```

| file | contents |
|---|---|
| `TabPFN_P_bubble_intervals.csv` | split, source_id, T, P, actual, mean, median, q0.0050 … q0.9950 |
| `TabPFN_P_bubble_coverage.json` | coverage table, feed map, test RMSE, reproduction check |
| `SVGP_gamma_intervals.csv` | split, source_id, T, P, actual, baseline, mean, lo/hi per level |
| `SVGP_gamma_coverage.json` | as above, plus `n_inducing` |
| `with_legend/`, `no_legend/` | `Figure11a-c` (TabPFN) and `Figure12a-c` (SVGP) at 1200 dpi |
| `Figure11_rebuttal.png`, `Figure12_rebuttal.png` | 3-panel montages embedded in the rebuttal .docx |

JSON width/error keys are **unit-neutral** (`mean_width`, `median_width`,
`test_rmse`) with the unit in a top-level `"unit"` field — bar for P<sub>bubble</sub>,
mN/m for γ.

### Reproduction checks

- **SVGP**: exact. Test RMSE 0.036609337701 vs the stored 0.0366093390147 — 8
  significant figures. Only the test split is compared; the validation rows
  carried along are just those of the example feeds, so their RMSE is not
  comparable with the stored full-split number.
- **TabPFN**: not bit-exact, and cannot be. The pickled preprocessing state in
  the archived `.joblib` does not survive the local environment (Snellius trained
  under Python 3.13 / sklearn 1.7.2; local is 3.12 / 1.6.1 —
  `ReshapeFeatureDistributionsStep` loses `subsampled_features_`), so the script
  rebuilds the estimator from `get_params()` and re-fits on the identical split,
  on CPU. Quantified in `reproduction_check`: run-to-run RMSE **0.0335 bar**
  against interval widths of 0.34–20 bar; feed-2 max deviation 0.509 bar (CPU) vs
  0.556 bar (archived H100); **all 61 archived H100 means fall inside the CPU
  95 % intervals**. Immaterial for this figure.

---

## Where it appears

| location | what |
|---|---|
| `Supporting-Information-A6/si.tex` §S4.3 | full explanation + Figures S26 and S27 (red, revision markup) |
| `A6-Draft-Overleaf/paper.tex` §3.4 | two sentences on the TabPFN intervals |
| `A6-Draft-Overleaf/paper.tex` §3.6 | two sentences on the SVGP intervals and the coverage contrast |
| `A6-Draft-Overleaf/paper.tex` §Supporting Information | one clause in the SI contents description |
| `Rebuttal_letter_A6.docx` | R2 point-4 response + both montages |

SI figure filenames track render order strictly, so each insertion forced a
renumber: adding S26 pushed the old S26–S32 to S27–S33, and adding S27 pushed
them again to S28–S34.
