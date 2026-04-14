# SVGP Residual Model — Overprediction Analysis & Fixes

## Component critical properties

| Component | $T_c$ / K | $P_c$ / MPa |
|-----------|----------:|------------:|
| CO₂       | 304.18    | 7.3825      |
| CO        | 134.45    | 3.49875     |
| H₂S       | 373.4     | 8.96291     |
| Ar        | 150.86    | 4.89805     |
| O₂        | 154.58    | 5.043       |
| N₂        | 126.19    | 3.3978      |
| CH₄       | 190.6     | 4.61        |
| H₂        | 33.18     | 1.300       |

---

## Observed problem

The parity plot for the SVGP residual model (`SVGP_gamma_parity_plot.png`) shows two
distinct failure modes:

1. **Underprediction of large positive residuals (regression-to-mean)**  
   For actual Δγ > ~2 mN/m the predicted values cluster *below* the diagonal. The
   model fails to reach the true magnitude of large positive residuals and instead
   pulls predictions back toward the bulk of the training distribution.

2. **Heteroscedastic fan at positive predictions**  
   The residual-vs-predicted plot (`SVGP_gamma_residual_vs_predicted.png`) shows
   prediction error variance increasing sharply for predicted Δγ > 1 mN/m, while the
   negative side is tight. This fan pattern is characteristic of a model fitted with a
   homoscedastic noise assumption on a skewed target.

---

## Root cause analysis

### Target distribution

```
count  41 696
mean   −0.376 mN/m
std     0.819 mN/m
25 %   −0.612 mN/m
50 %   −0.222 mN/m   ← median is negative
75 %   −0.051 mN/m
max    +6.967 mN/m
```

The distribution is **strongly left-skewed**: more than 75 % of samples carry a
residual between −0.61 and −0.05 mN/m. Large positive residuals (> 1 mN/m) are a
rare, heavy tail that represents the physically interesting high-interface-tension
regime.

### Why the SVGP struggles here

| Mechanism | Effect |
|-----------|--------|
| **k-means inducing point init** | Places ≈95 % of the 800 inducing points in the dense negative region; the positive tail receives almost no inducing points. SVGP predictions in underrepresented regions revert to the prior mean → systematic underprediction. |
| **Homoscedastic `GaussianLikelihood`** | A single global noise parameter fits the tight bulk. Tail samples receive the same (small) noise, so their high variance looks like signal. The model cannot distinguish high-variance-due-to-scarcity from high-variance-due-to-physics. |
| **Symmetric RBF kernel** | Gaussian (squared-exponential) kernel decays very quickly away from training density. In the tail the covariance falls to near-zero, again reverting predictions to the prior. A Matérn kernel has heavier tails in function space and generalises better from sparse data. |
| **Skewed target passed to ELBO** | StandardScaler normalises to zero-mean/unit-variance but does not remove skewness. The ELBO loss is dominated by the 75 % of samples with small negative residuals, which outweigh the informative tail samples. |

---

## Results after round 1 fixes

| Metric | Baseline | After round 1 |
|--------|----------|---------------|
| Train R² | 0.98 | **0.99** |
| Test R²  | 0.98 | 0.98 |
| Val R²   | 0.98 | **0.99** |

Test R² did not improve despite train/val gains — the ELBO gradient was still dominated
by the bulk (~75 % negative-residual samples), so tail learning stalled on unseen test
points.

---

## Round 2 — further tail improvement

### Remaining problem

The positive tail (actual Δγ > 2 mN/m) still shows a scatter fan on the test set.
Even with the sign-log transform and 20 % tail inducing points, tail samples contribute
only ~5 % of each mini-batch's ELBO gradient. The optimiser barely "sees" them per
epoch.

### Additional fixes (`svgp_02_config`, `svgp_10_init`)

| Parameter | Round 1 | Round 2 | Reason |
|-----------|---------|---------|--------|
| `N_INDUCING` | 1 200 | **1 500** | More total capacity |
| `N_INDUCING_TAIL` | 0.20 | **0.30** | 450 tail pts instead of 240 |
| `TAIL_THRESHOLD` | 1.0 mN/m | **0.5 mN/m** | Captures moderate tail too |
| `TAIL_OVERSAMPLE` | — | **3×** | New: tail mini-batch share ~25 % |

**WeightedRandomSampler** (new in `svgp_10_init`):

```python
sample_weights = np.where(np.abs(y_train) > TAIL_THRESHOLD, 3.0, 1.0)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
```

Tail samples (|residual| > 0.5 mN/m) are drawn 3× more often than bulk samples,
raising their effective mini-batch share from ~5 % to ~25 %. This is the most direct
way to increase gradient signal from the extremes without changing the model
architecture or loss function.

Replacing `shuffle=True` with the weighted sampler makes the `generator` seed in the
old `DataLoader` unnecessary; the `Generator` is instead passed to
`WeightedRandomSampler` for reproducibility.

---

## Changes implemented in `SVGP_GPR_RESIDUAL.ipynb`

### 1. Stratified inducing point initialisation  (`svgp_09_model_def`, `svgp_10_init`)

`init_inducing_points` was rewritten with a two-stage scheme controlled by two new
config variables (`N_INDUCING_TAIL = 0.20`, `tail_threshold = 1.0 mN/m`):

```python
n_tail = int(round(N_INDUCING * N_INDUCING_TAIL))   # 240 points
n_bulk = N_INDUCING - n_tail                         # 960 points

bulk_pts = MiniBatchKMeans(n_clusters=n_bulk).fit(X_scaled).cluster_centers_
tail_idx = rng.choice(np.where(np.abs(y_raw) > tail_threshold)[0], size=n_tail)
tail_pts = X_scaled[tail_idx]

inducing_points = np.vstack([bulk_pts, tail_pts])
```

- **960 bulk points** from k-means cover the dense negative region.
- **240 tail points** are randomly drawn from samples where `|residual| > 1 mN/m`,
  giving the posterior direct anchors in the high-tension extremes.

### 2. Matérn 5/2 kernel replaces RQ kernel  (`svgp_09_model_def`)

```python
# Before
self.covar_module = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.RQKernel(ard_num_dims=d)
)
# After
self.covar_module = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)
)
```

Matérn 5/2 is twice-differentiable (same smoothness class as RBF/RQ) but has
**polynomial** rather than Gaussian or power-law covariance decay. In sparse regions
the covariance stays non-negligible for longer, so predictions do not snap to the prior
mean as sharply.

### 3. Sign-log target transform  (`svgp_08_preprocess`, `svgp_13_predict`)

Applied before `StandardScaler` normalisation:

```python
def signlog(x):     return np.sign(x) * np.log1p(np.abs(x))
def signlog_inv(x): return np.sign(x) * np.expm1(np.abs(x))

y_train_sl = signlog(y_train.values)   # fit scaler on this
```

`sign(y) · log(1 + |y|)` compresses the heavy positive tail symmetrically while
preserving sign and the zero-crossing. The ELBO loss then treats the bulk and tail more
equitably. At prediction time the pipeline is reversed:

```
raw normalised output
  → × y_std + y_mean          (undo StandardScaler, still in sign-log space)
  → signlog_inv(·)             (back to mN/m)
```

Uncertainty (std) is propagated through the linearised inverse:
`σ_mN/m ≈ σ_norm · y_std · exp(|ŷ_sl|)`.

### 4. Hyperparameter updates and cosine LR annealing  (`svgp_02_config`, `svgp_10_init`, `svgp_11_train`)

| Parameter | Before | After |
|-----------|--------|-------|
| `N_INDUCING` | 800 | 1 200 |
| `N_EPOCHS` | 100 | 200 |
| LR schedule | constant 0.01 | `CosineAnnealingLR(T_max=200, eta_min=1e-4)` |

`CosineAnnealingLR` smoothly decays the learning rate from 0.01 → 1×10⁻⁴ over
training, avoiding the late-epoch loss oscillation that occurs with a fixed LR and
allowing the variational parameters to settle into a sharper minimum.

`scheduler.step()` is called once per epoch (after all mini-batch steps), and the
current LR is logged every 10 epochs.

### 5. Model checkpoint updated  (`svgp_22_save_model`)

The saved `.pt` file now includes `"y_transform": "signlog"` so any downstream
inference code can reconstruct the full pipeline without inspecting the notebook.

---

## Expected outcome after round 2

| Metric | Baseline | Round 1 | Round 2 target |
|--------|----------|---------|----------------|
| Train R² | 0.98 | 0.99 | ≥ 0.99 |
| Test R²  | 0.98 | 0.98 | **≥ 0.985** |
| Val R²   | 0.98 | 0.99 | ≥ 0.99 |
| Tail scatter (actual > 2 mN/m) | wide fan | moderate fan | tight, near diagonal |
| Residual-vs-predicted fan | opens at Δγ > 1 | reduced | substantially closed |
| Reconstructed γ R² | 0.9995 | ~0.9996 | ≥ 0.9997 |

The reconstructed parity (gamma_base + residual) benefits directly from tail
improvement — the high interface-tension regime (large positive residuals) is where the
base model's WSD approximation is least accurate, making residual prediction most
physically meaningful there.

---

## Round 3 — candidate improvements (not yet implemented)

### Candidate 1 — QuantileTransformer replaces sign-log

`sklearn.preprocessing.QuantileTransformer(output_distribution='normal',
n_quantiles=min(n_train, 1000))` maps the empirical CDF to a standard normal.
Every quantile contributes equally to the ELBO loss, so tail samples receive the same
weight per sample as bulk samples — stronger than sign-log which only partially
compresses the tail.

Std propagation uses the linearised Jacobian of the inverse CDF:
`σ_mN/m ≈ σ_norm / pdf_train(ŷ_mN/m)`, evaluated numerically from the fitted
transformer.

The transformer must be fit on `y_train` only and applied before `StandardScaler`.

### Candidate 2 — Increase `TAIL_OVERSAMPLE` 3× → 5×

Low-risk, single-line change. Combined with the quantile transform, tail samples will
appear more frequently *and* occupy a more favourable part of the transformed target
space.

### Implementation order

1. Replace sign-log with QuantileTransformer (candidate 1) — run and evaluate parity plot.
2. If tail scatter persists, increase TAIL_OVERSAMPLE to 5 (candidate 2) as a final tuning step.
