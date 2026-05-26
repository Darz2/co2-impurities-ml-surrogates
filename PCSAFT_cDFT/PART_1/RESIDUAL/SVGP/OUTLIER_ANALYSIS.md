# SVGP Residual — Worst-Outlier Analysis

**Notebook:** [SVGP_GPR_RESIDUAL.ipynb](SVGP_GPR_RESIDUAL.ipynb)
**Point under inspection:** `source_id = 21`, `T = 200 K`, `P = 161.246 bar`, `residual_actual = -4.986 mN/m`

---

## 1. Notebook Diagnostic Output

From the nearest-neighbour diagnosis cell (`12f07684`):

| Metric | Value |
|---|---|
| Predicted residual | −3.429 mN/m |
| Predictive std (σ) | 0.778 mN/m |
| Absolute error | 1.558 mN/m (≈ 2σ) |
| Nearest training-neighbour distance (scaled) | **0.395** |
| Neighbour residual mean / std | −3.673 / 0.646 |
| Neighbour residual range | [−4.615, −2.688] |
| Automatic verdict | **FEATURE-SPACE CONFLICT** |

Top-8 nearest training neighbours are *all* `source_id = 21`, with residuals progressively
shrinking in magnitude as we move away from the outlier:

| Rank | Scaled dist | Neighbour residual |
|---|---|---|
| 1 | 0.395 | −4.406 |
| 2 | 0.776 | −4.129 |
| 3 | 1.118 | −3.878 |
| 4 | 2.002 | −3.221 |
| 5 | 2.061 | −4.152 |
| 6 | 2.260 | −3.029 |
| 7 | 2.488 | −2.855 |
| 8 | 2.707 | −2.688 |

So the point is **not isolated** — the model has nearby anchors. It misses the value because
the residual surface keeps bending steeper *past* the last training anchor, and a stationary
GP cannot follow it.

---

## 2. Why the ML Could Not Learn This Point

### 2.1 Extreme-tail scarcity
Only **81 training samples** have `|residual| ≥ 3.146 mN/m` (cell `svgp_10_init` raised
`[warn] only 81 extreme-tail samples; falling back to random draw`). The full training set
is 16 130 points, so the absolute tail is < 0.5 % of the data. The point at −4.99 sits at
the *outermost* edge of this already-thin tail.

### 2.2 Stationary kernel cannot bend sharply enough
The Matern-5/2 ARD kernel has **global** length-scales. Looking at the neighbours, the
function decreases from −3.88 → −4.13 → −4.41 over a scaled distance of ~1.1, and is
expected to continue to −4.99 over the next ~0.4. That curvature is sharper than the
fitted length-scale can support, so the GP posterior shrinks the prediction toward the
local conditional mean (≈ −3.67), giving the observed −3.43.

### 2.3 QuantileTransformer compresses the tail
`QuantileTransformer(output_distribution="normal")` maps the empirical CDF onto N(0, 1).
This is great for the bulk but pushes the most-extreme samples into the asymptotic tail of
the normal, where the **inverse-transform Jacobian (`qt_jacobian`) is very large**. A modest
under-prediction in normalised space therefore amplifies into ~1.5 mN/m of error in the
original units. With only 81 extreme samples, the empirical quantile estimate of the tail
is also noisy.

### 2.4 Inducing-point starvation at the extreme corner
The three-tier init requested 150 extreme-tail inducing points but only 81 such samples
existed, so it fell back to a random draw. The variational posterior at this corner of
feature space is consequently under-parameterised.

### 2.5 The model knows it is uncertain
`abs_error (1.558) ≈ 2 × σ_pred (0.778)`. The predictive interval already covers the gap —
calibration is fine. The model is *honest* about not knowing, but it cannot break through
the smoothness prior.

### 2.6 Likely physical reason
(T = 200 K, P ≈ 161 bar) is the **low-T / high-P corner** for `source_id = 21`, almost
certainly near a phase boundary or near-critical regime where `cDFT − WSD` diverges.
That is exactly where the residual should look near-singular, and exactly where a smooth
stationary GP prior is structurally the wrong inductive bias.

---

## 3. Deep Dive — Why "only 81 samples with |y| ≥ 3.146"?

### 3.1 Where the threshold 3.146 mN/m comes from

In cell `svgp_10_init`:

```python
extreme_thresh = np.percentile(y_train_abs, 99.5)
```

It is the **99.5th percentile of |residual|** in the training set. By definition, exactly
0.5 % of training samples lie beyond it:

```
16 130 training samples  ×  0.5 %  ≈  80.65   →   81 samples
```

So the count is mechanically forced by the percentile choice. The real question is what the
underlying residual distribution looks like:

| Bucket          | Threshold                  | Count   | Share of train |
|-----------------|----------------------------|--------:|---------------:|
| Bulk            | \|y\| ≤ 0.3 mN/m            | ~8 269  | ~51 %          |
| Mid-tail        | 0.3 < \|y\| < 3.146         | 7 780   | ~48 %          |
| **Extreme tail**| **\|y\| ≥ 3.146**           | **81**  | **0.5 %**      |

The residual `gamma_cDFT_UC − gamma_wsd_UC` is heavily concentrated near zero — across
~99.5 % of (T, P, composition) space the two models agree to within ~3 mN/m. Only a few
specific corners (low T, high P, near phase boundaries) push the magnitude beyond that. The
distribution is **heavy-tailed and geometrically narrow** in feature space.

### 3.2 Why this breaks the stratified inducing-point initialisation

The init step asked for 10 % of `N_INDUCING = 1500` to come from the extreme tail:

```python
n_extreme = round(N_INDUCING * extreme_frac) = round(1500 * 0.10) = 150
```

…but only **81 candidate samples** exist, so the warning fires:

```
[warn] only 81 extreme-tail samples; falling back to random draw
```

…and the 150 extreme inducing points are filled in with **random points from anywhere in
feature space**, not from the tail. The deliberate stratification breaks down precisely where
it was meant to help most.

### 3.3 Why few anchors hurt an SVGP more than a vanilla GP

- **Inducing points are the resolution of an SVGP.** The variational posterior is
  parameterised at the inducing locations and interpolated between them. Where inducing
  points are dense, the model can represent fine structure; where they are sparse, it
  collapses onto the kernel's smoothness prior.
- **The extreme corner has at most 81 candidate anchors**, scattered across whatever
  combinations of (T, P, composition) produce `|residual| ≥ 3.146`. Even using all 81 as
  inducing points (you cannot — some are near-duplicates in feature space) gives at most
  ~1 anchor per very-different physical corner.
- **The function in this corner is also the steepest** (residual moves from −2.7 → −4.6
  across the worst-outlier's nearest neighbours within scaled distance ~2). The model needs
  **more** resolution in the tail but the data supplies **less**.

### 3.4 Three compounding effects from the same scarcity

1. **Sparse anchors** — only 81 candidates for a region that demands the highest resolution.
2. **Noisy QuantileTransformer tail** — `QuantileTransformer` estimates the empirical CDF;
   with only 81 samples beyond `|y| = 3.146`, the inverse-transform Jacobian at the
   asymptote is fit from very little data. Small ML errors near the QT-N(0,1) tail get
   inflated when mapped back to mN/m.
3. **Weighted-sampler ceiling** — the `WeightedRandomSampler` gives extreme samples
   `3 × 5 = 15×` the draw probability, but oversampling 81 unique points 15× **does not
   create new information**; it just shows the same 81 points more often. The gradient
   signal is bounded by the information those 81 carry.

### 3.5 What actually fixes it

- **Generate more training data in the extreme corner** — densify the cDFT/WSD simulations
  on the (T, P, composition) corners that produce `|residual| > 3` (predominantly
  `source_id = 21, 50, 79` etc., low-T high-P points). This is the only fix that adds
  *information*.
- **Lower the extreme percentile to 99 %** → ~161 candidates instead of 81. Buys more
  anchors but cannot manufacture data the simulations never produced.
- **Stratified KFold by `|y|` bin** — confirm the 81 tail samples are spread across folds
  rather than bunched in one.
- **Non-stationary kernel / heteroscedastic likelihood** — lets the model bend sharper near
  phase boundaries without needing more inducing anchors. Mitigates symptom 1 but not 2 or 3.

> **Headline.** 81 isn't too few because of a code choice — it's too few because the
> underlying residual distribution is heavy-tailed and the simulations that produced large
> residuals weren't sampled densely enough. **The real fix is upstream of the ML.**

---

## 4. Summary

| Possible cause | Verdict |
|---|---|
| Isolated in feature space (no training anchor) | **No** — nearest neighbour at 0.395 |
| Too few training samples in the extreme tail | **Yes** — only 81 with \|y\| ≥ 3.146 |
| Stationary kernel too smooth to follow the gradient | **Yes** — dominant cause |
| QT inverse Jacobian amplifies tail errors | **Yes** — contributes to the magnitude |
| Inducing-point coverage at the extreme | **Yes** — fell back to random draw |
| Numerical anomaly in the underlying cDFT calculation | **Possible** — flagged by automatic verdict |
| Model is mis-calibrated here | **No** — 2σ already covers the gap |

---

## 5. Recommended Next Steps

1. **Verify the cDFT computation** for `source_id = 21` at `(T = 200 K, P = 161.25 bar)`.
   The automatic verdict (`FEATURE-SPACE CONFLICT`) is consistent with a near-critical /
   phase-boundary numerical artefact in the source data.
2. **Densify training data** around this corner for `source_id = 21` rather than relying on
   oversampling the 81 existing extreme points.
3. **Try a non-stationary kernel** (e.g. input-dependent length-scale) or a
   **heteroscedastic likelihood** so the model can bend sharply near phase boundaries.
4. **Gate predictions by predictive uncertainty** — treat points with `pred_std > τ` as
   "out of trust region" rather than reporting a point estimate.
