# GPR Residual Model — Scientific Discussion

---

## 1. Why Standard GPR Cannot Scale to 40,000 Points

Standard `sklearn` GaussianProcessRegressor requires building and factorising an **n × n kernel matrix** (Cholesky decomposition) — both time and memory scale poorly:

| Samples | Kernel matrix (float64) | Training time | Verdict |
|---|---|---|---|
| ≤ 1,000 | < 8 MB | seconds | comfortable |
| 2,000 | ~32 MB | minutes | current GPR setup |
| 5,000 | ~200 MB | ~30–60 min | pushing it |
| 10,000 | ~800 MB | hours | borderline |
| **40,000** | **~12.8 GB** | **days** | **infeasible** |

The bottleneck is unavoidable in exact GPR: O(n³) time and O(n²) memory. For the 40k dataset, `EXPERIMENT_MAX_SAMPLES = 2000` is not a workaround — it is the correct operating range for this algorithm.

---

## 2. Sparse GP vs Variational GP (SVGP)

Both approaches use **inducing points** — a small set of m representative pseudo-inputs (m ≪ n) — to approximate the full GP, but differ in how the approximation is made.

### Sparse GP (FITC / Nyström)

- **Idea:** Approximate the full covariance matrix as K ≈ Kₙₘ Kₘₘ⁻¹ Kₘₙ using only m inducing points.
- **Training:** Exact inference on the compressed representation; optimises the *approximate* marginal likelihood.
- **Complexity:** O(nm²) time, O(nm) memory — large saving when m ~ 500.
- **Weakness:** Inducing point locations can cluster; can be brittle on non-stationary data.
- **Example:** `GPy` library (FITC/PITC kernels).

### Variational GP (SVGP)

- **Idea:** Treat the GP posterior as something to be *learned* by minimising KL-divergence from the true posterior. Inducing points and their values are **variational parameters**.
- **Training:** Optimises an Evidence Lower BOund (ELBO) via stochastic mini-batch gradient descent — analogous to neural network training.
- **Complexity:** O(m³) per step, O(m²) memory — scales to millions of points.
- **Strength:** Mini-batching means the full n × n matrix is never formed; memory footprint is fixed at m regardless of dataset size.
- **Example:** `GPyTorch` (SVGP), `GPflow`.

### Side-by-Side

| | Sparse GP (FITC) | Variational GP (SVGP) |
|---|---|---|
| Inference | Exact (on approximation) | Approximate (ELBO optimisation) |
| Training | L-BFGS on full data | SGD with mini-batches |
| Memory | O(nm) — still scales with n | O(m²) — fixed, independent of n |
| Uncertainty | Calibrated | Calibrated (if ELBO converged) |
| Scalability | ~100k (practical ceiling) | Millions of points |

---

## 3. SVGP Design Decisions for This Dataset

| Component | Choice | Reason |
|---|---|---|
| Inducing points | `MiniBatchKMeans` centroids (m = 500) | Better T–P–z coverage than random selection; fast on 40k |
| Kernel | `ScaleKernel(RBFKernel(ard_num_dims))` | Same ARD structure as the sklearn GPR — one length-scale per feature |
| y normalisation | Manual mean/std | Equivalent to `normalize_y=True` in sklearn GPR |
| Predictions | Batched (to avoid OOM on 40k) | GPyTorch posterior on 40k points at once would exceed RAM |
| sklearn wrapper | `SVGPWrapper` | Gives a `predict(X, return_std=True)` interface so all `MLPostprocessing` plots work unchanged |
| CV | Manual `KFold` loop | `cross_val_score` does not support GPyTorch estimators |

**Memory comparison:**

| | sklearn GPR | SVGP |
|---|---|---|
| Kernel matrix | O(n²) — 12.8 GB at n = 40k | O(m²) — ~1 MB at m = 500 |
| Training time | O(n³) — infeasible | O(nm²) per epoch — feasible |
| SLURM memory | 16G × 4 CPU | 8G × 4 CPU |

Why memory stays fixed during mini-batch training: each SGD step loads only a **batch** of b points (b ≪ n), computes the ELBO gradient using only the m inducing points, and discards the batch. The kernel matrix that is actually built at each step is b × m, not n × n — so peak memory is O(bm), independent of n.

---

## 4. SVGP Historical Context

SVGP is well-established, not experimental:

| Year | Milestone |
|---|---|
| 2009 | Sparse GP / inducing points formalised (Snelson & Ghahramani; Titsias) |
| **2013** | SVGP paper: *"Gaussian Processes for Big Data"* — Hensman, Fusi, Lawrence |
| 2015 | Extended to non-Gaussian likelihoods (classification) — Hensman et al. |
| 2017 | GPyTorch released, making SVGP practical and fast |
| 2021 | GPyTorch 1.x — now the standard library for scalable GPs |

The core algorithm is ~12 years old (2013). More recent work (2020–2024) builds *on top* of SVGP (deep kernel learning, multi-output GPs, natural gradient variational inference) but the base algorithm is unchanged. For thermodynamic data at this scale requiring calibrated uncertainty on γ_cDFT residuals, SVGP is the appropriate and publishable choice.

---

## 5. 2D T–P Response Surface

### Reading the Plot

**Axes:** Temperature T (K) on x, Pressure P (bar) on y.
**Colour:** Predicted mean residual Δγ = γ_cDFT − γ_base in mN m⁻¹ (RdBu_r: blue = negative, red = positive, white ≈ 0).
**Dots:** Training-data scatter, coloured by actual y_train values — showing where the model learned vs. where it extrapolates.

### Physical Interpretation

| Region | T, P | Δγ | Meaning |
|---|---|---|---|
| **Deep blue** | Low T (200–230 K), moderate-high P (40–120 bar) | −2 to −4 mN/m | cDFT surface tension is well **below** the WSD baseline; the CO₂-density-gap correction over-predicts γ at these conditions |
| **White band** | Diagonal crossing ~240–260 K | ≈ 0 | Baseline and cDFT agree; the residual the GPR must correct is near zero |
| **Red** | High T (>270 K), low P (<30 bar) | +1 to +2.4 mN/m | cDFT is **above** the baseline; baseline under-predicts γ in the near-critical / dilute region |

### Key Structural Features

1. **Curved zero-crossing band** runs diagonally from upper-left to lower-right — the sign of the correction flips as you move from dense/cold to hot/dilute conditions.
2. **Strong T-sensitivity at low P** — the colour gradient is steepest along the x-axis below ~40 bar, meaning T drives the residual more than P in the dilute regime.
3. **P-sensitivity dominates at low T** — the tight blue contours at 200–230 K show that compressing the mixture (raising P) amplifies the negative correction.
4. **Sparse coverage at high T / low P** (~260–300 K, <20 bar) — training data is thin there; the red prediction in that region is extrapolation and should be treated with more caution.

### In Context of the Model

The GPR learned that γ_base = γ₀_CO₂ · r₁² systematically **overestimates** γ_cDFT in the cold-dense quadrant and **underestimates** it in the hot-dilute quadrant. The 2D surface is the correction function the model applies before reconstructing the full γ_cDFT.

---

## 6. Parity Plot — Identifying Outliers

The two extreme outlier points visible in the parity plot (validation set, red circles, actual Δγ ≈ +4–5 mN/m, predicted ~+2 mN/m) can be traced back to their source via `source_id` and dominant co-solvent species (largest z_* after CO₂).

---

## 7. Δγ vs T Scatter — Baseline Over/Underprediction

### What the Plot Shows

This is the **raw target** Δγ = γ_cDFT − γ_base vs T, coloured by P. The over/underprediction shown here is by the **baseline** (γ_base = γ₀_CO₂ · r₁²), not the GPR model.

| Sign | Meaning |
|---|---|
| Points **above** zero (Δγ > 0) | Baseline **underpredicts** γ_cDFT — GPR must add a positive correction |
| Points **below** zero (Δγ < 0) | Baseline **overpredicts** γ_cDFT — GPR must subtract |

### Key Patterns

- **Low T (200–230 K):** widest spread in both directions — the baseline is least reliable here; both signs occur, likely driven by mixture composition (blue = low P points dominate).
- **One red point at T ≈ 200 K, Δγ ≈ −6 mN/m** (high P, ~120 bar) — extreme negative outlier; the baseline strongly overpredicts at very cold, compressed conditions.
- **High T (>260 K):** data collapses toward zero — baseline works well in the warm, dilute regime.
- **Asymmetry:** more points below zero than above — the baseline systematically overpredicts on average (consistent with residual statistics: mean Δγ ≈ −0.41 mN/m).

### Implication for GPR

The GPR must learn a **non-trivial sign-changing function** of T and P — exactly what the 2D response surface confirms: negative corrections at low T/high P, positive at high T/low P.

---

## 8. Stratified T–P Sampling

### Motivation

The original notebook used `df.sample(n=2000, random_state=SEED)` — pure random sampling. With ~40k total rows, rare T–P regions (very low T, very high P) are often missed entirely, causing the train/val R² gap (train 0.99, test 0.98, val 0.91). The validation split lands in regions the training set barely saw.

### How Stratified Sampling Works

#### Step 1 — Divide T into 10 quantile bins

`pd.qcut(df["T"], q=10)` creates **deciles** — each bin contains roughly the same number of rows. The bin edges are not evenly spaced; they are denser where data is dense. This is important: if most data clusters near 300 K, the 200–230 K range gets its own early bins with guaranteed representation.

#### Step 2 — Divide P into 10 quantile bins

Same approach along the pressure axis.

#### Step 3 — Cross the two → up to 100 strata

Each stratum is a **T-band × P-band cell**. A row with T-bin=2, P-bin=7 gets label `"2_7"`. In practice some cells may be empty (e.g. very low T + very high P may not exist), so the actual number of strata is ≤ 100.

#### Step 4 — Sample proportionally from each stratum

$$n_{\text{stratum}} = \max\!\left(1,\ \text{round}\!\left(\frac{|\text{stratum}|}{|\text{full dataset}|} \times N_{\text{target}}\right)\right)$$

- A dense stratum → draws more points (proportional to its share)
- A **rare** stratum → draws at least **1 point**, guaranteed
- The `max(1, ...)` is the critical difference from random sampling — rare cells can never be skipped

#### What is NOT Stratified

Composition (z_*) is not used for binning. The assumption is that T and P drive the structural gaps in the residual, and composition varies naturally within each T–P cell.

### Before vs After (intuition)

```
Random sampling (old)         Stratified sampling (new)

P ▲                           P ▲
  │  ···· ····                  │  · · · ·
  │  ···· ····                  │  · · · ·
  │       ····                  │  · · · ·
  │            ··               │  · · · ·
  └──────────► T                └──────────► T

Dots cluster where data        Dots spread across the
is dense — sparse corners      full T–P envelope
get nothing
```

The extreme outliers (very low T, very high P) were in a sparse corner. Random sampling often missed them entirely; stratified sampling guarantees at least one draw from that T–P cell, so the model sees and learns those conditions.

---

## 9. Stratified Split — Extending to Dominant Impurity Species

### Motivation

Stratifying only on T and P was not enough. Certain impurity species (notably H₂S) are represented in very specific T–P windows. With a 2D T–P stratum, splits could still place all H₂S samples in the test set simply because they occupy a T–P cell that happened to be assigned there. The result: train R² ≈ 0.99, test/val R² ≈ 0.92 — the same train/test gap as before.

### Implementation

The stratum label is extended to a **3D key**: T-bin × P-bin × dominant_impurity.

- **Dominant impurity** = `idxmax` of all non-CO₂ z_* columns (CO₂ is always the dominant species, so the "second highest" component is what distinguishes mixture type). Six distinct impurity labels in this dataset: Ar, CO, H₂, H₂S, CH₄, N₂.
- **Bin counts:** `STRAT_BINS_SPLIT = 2` → 2×2×6 = 24 strata (minimum ~4 members per stratum at n = 1000, well above sklearn's minimum-2 requirement).
- `STRAT_BINS_SAMPLING = 10` is kept for the initial random draw (singletons are acceptable in sampling; sklearn `stratify=` is not used there).

---

## 10. ARD-RBF ConvergenceWarning — Sparse Composition Features

### What Happens

The ARD-RBF kernel assigns one **length scale per feature**. A large length scale means "this dimension is irrelevant — the kernel is insensitive to variation along it." When a feature has near-zero variance in the training set (essentially always zero), the GPR optimizer correctly concludes that the optimal length scale is infinity, and keeps pushing it toward the upper bound until it hits the constraint.

This triggers a `ConvergenceWarning` from sklearn:

```
ConvergenceWarning: lbfgs failed to converge (status=2):
ABNORMAL_TERMINATION_IN_LBFGS
```

and produces **inflated predictive uncertainty** (blown-out ±2σ error bars) because the hyperparameter optimum was not truly reached.

### Why Raising the Bound Does Not Help

| Bound | Result |
|---|---|
| `(1e-2, 1e3)` | Hits `1e3`, ConvergenceWarning |
| `(1e-2, 1e5)` | Hits `1e5`, still ConvergenceWarning |
| `(1e-2, 1e10)` | Hits `1e10`, same warning |

The true optimum is ∞. No finite upper bound will prevent the warning — the optimizer gradient at the boundary is non-zero, so L-BFGS never sees a true minimum.

### The Root Cause

Composition features like `z_carbon_monoxide`, `z_methane`, and `z_nitrogen` are sparse: they are zero for the vast majority of data points (most mixtures in this dataset do not contain those components). In a random subsample of 1000–2000 points, these features may have a training-set standard deviation below 0.01 — effectively zero for the purposes of the kernel.

### The Fix — Variance-Threshold Feature Selection

After the train/test/val split, compute the standard deviation of each feature **on X_train only** (no data leakage), and drop any feature below a threshold:

```python
_train_std = X_train.std()
_keep      = _train_std[_train_std >= VAR_THRESHOLD_STD].index.tolist()
X_train, X_test, X_val = X_train[_keep], X_test[_keep], X_val[_keep]
features = _keep
```

`VAR_THRESHOLD_STD = 0.01` is the global parameter (exposed to papermill). Features dropped at n = 1000 typically include `z_carbon_monoxide`, `z_methane`, and `z_nitrogen`; at n = 5000 the set stabilises as more rare mixture types are included in the sample.

**Why this is correct:** a feature the GPR cannot learn anything from (because it never varies in training) should not be in the model. Excluding it removes an unsolvable optimisation dimension and gives the L-BFGS clean convergence on the remaining features.
