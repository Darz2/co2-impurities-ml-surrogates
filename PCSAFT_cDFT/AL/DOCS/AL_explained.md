# Active Learning — Full Explanation

## 1. Overall Pipeline

The AL directory contains four notebooks that together implement and compare three sampling strategies on a 1000-composition pool of CO2-rich CCS mixtures.

```
01_generate_pool.ipynb      →  generates 1000 compositions (the pool)
02_Random_Sampling.ipynb    →  baseline: pure random draws from the pool
03_Stratified_Sampling.ipynb →  structured: CO2 bin × K-means impurity cluster
02_GPR_AL.ipynb             →  active learning: GPR kernel uncertainty (Algorithm 1)
04_AL_Sampling.ipynb        →  active learning: label-free RBF greedy selection
```

**Pool composition** (`01_generate_pool.ipynb`):
- 1000 compositions, 8 species: CO2, H2, Ar, N2, CH4, O2, CO, H2S
- CO2 in range 0.95–1.0
- Split: 70% random Dirichlet, 20% systematic, 10% industrial templates
- All zero kij pairs (same convention as DATASET_A1)

---

## 2. Stratified Sampling (`03_Stratified_Sampling.ipynb`)

### The Goal

Draw a representative subset of size N from the pool. Random sampling can accidentally over-represent one CO2 level or one impurity type. Stratification forces coverage across the full composition space.

The strata form a **2D grid**: CO2 level × impurity pattern = 25 boxes. You proportionally sample from each box.

### Step 1 — CO2 Binning

```python
CO2_BINS   = [0.95, 0.96, 0.97, 0.98, 0.99, 1.001]
CO2_LABELS = ["0.95-0.96", "0.96-0.97", "0.97-0.98", "0.98-0.99", "0.99-1.00"]
pool["co2_bin"] = pd.cut(pool["CO2"].round(4), bins=CO2_BINS, labels=CO2_LABELS, right=False)
```

Result: ~180–245 compositions per bin.

### Step 2 — Per-Bin Impurity Clustering (K-means)

**Why normalize by total impurity first?**

At 0.99 CO2 the impurities are ~1% total; at 0.95 CO2 they are ~5%. Clustering raw mole fractions conflates "how much impurity" with "what type of impurity." Normalizing separates the two:

```python
X_imp_norm = x_impurity_species / (1.0 - CO2)   # each row sums to ~1
```

Now each row is a composition of the impurity fraction only — e.g. "30% N2, 20% H2, 50% CH4."

**Per-bin procedure:**
1. Normalize impurity fractions as above
2. Apply `MinMaxScaler` (fit per bin, not globally)
3. Run `KMeans(n_clusters=5)` on the scaled vectors

Done separately per CO2 bin because the range and spread of impurity patterns differs between bins.

**Result:** 5 clusters × 5 bins = **25 strata** total.

```
imp_cluster   0   1   2   3   4
co2_bin
0.95-0.96    42  41  44  55  63
0.96-0.97    27  25  62  33  44
0.97-0.98    40  39  26  66  28
0.98-0.99    38  22  28  59  36
0.99-1.00    33  48  36  28  37
```

Each composition gets a label: `stratum = co2_bin + "__c" + imp_cluster`.

### Step 3 — What the PCA Plot Is (and Isn't)

The PCA plot is **only for visualization**, not part of the clustering:
1. Takes all 1000 normalized impurity vectors
2. Applies a **global** MinMaxScaler (different from the per-bin scalers used for clustering)
3. Projects to 2D with PCA
4. Colors each point by its K-means cluster label

The PC axes explain variance in impurity-type space. Clusters appear as rough blobs — not perfectly separated because the actual clustering was done per-bin on a different scaled space.

### Step 4 — Proportional Allocation + Sampling

```python
def proportional_allocate(counts, N):
    alloc = (counts / counts.sum() * N).round().clip(lower=1).astype(int)
    while alloc.sum() > N: alloc[alloc.idxmax()] -= 1
    while alloc.sum() < N: alloc[alloc.idxmin()] += 1
    return alloc
```

For each of 20 trials (different RNG seed), sample `alloc[stratum]` compositions without replacement from each of the 25 boxes and concatenate.

**Output:** 100 CSV files — `N020/trial_00.csv` through `N100/trial_19.csv`.

### Comparison: Stratified vs Random

The last cell re-assigns random samples to the per-bin K-means clusters via `predict_bin_clusters()` (uses saved `scalers_per_bin` and `kmeans_per_bin` models) and plots side by side:
- **Top row:** CO2 bin distribution — stratified is more uniform
- **Bottom row:** impurity cluster distribution — stratified forces ~equal coverage

---

## 3. Active Learning — Algorithm 1 (`02_GPR_AL.ipynb`)

### The Core Idea

Instead of labeling all 1000 pool compositions, find the **smallest subset** that spans the composition space well — so a GPR trained on it generalizes as if you'd used the full pool. The algorithm does this **without any labels** — it uses only the GPR kernel (a measure of compositional similarity) to decide which point to add next.

### Phase 1 — Fit GPR Hyperparameters on A2 Data

```
A2 dataset (labeled: z → P_bubble at T=250 K)
    ↓
GPR: Matérn-5/2 ARD kernel + WhiteKernel
    ↓
Fitted kernel with ARD length-scales per component
```

**Kernel structure:**
```python
kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))            # σ_f²  — overall signal amplitude
    * Matern(nu=2.5,
             length_scale=np.ones(n_feat),       # one l_i per feature (ARD)
             length_scale_bounds=(1e-3, 1e3))
    + WhiteKernel(noise_level=0.1, ...)          # σ_n²  — observation noise
)
```

After fitting, the **WhiteKernel is stripped** — only the signal kernel (`ConstantKernel × Matérn`) is kept for the AL loop. The noise term should not contribute to the uncertainty calculation.

### Phase 2 — Algorithm 1 (Xiang et al., JCIM 2023)

```
S ← 2 random compositions from pool      (initial "selected" set)
P ← pool \ S                             (remaining "candidates")

while |S| < 100 and P is not empty:

    T ← random subset of P (or all of P if N_t = None)

    U_T = diag( K_TT  −  K_TS · K_SS⁻¹ · K_ST )   ← kernel uncertainty

    α = argmax(U_T)                       ← most "surprising" point
    if U_T[α] > u_t (=1e-6):
        S ← S ∪ {α}
        P ← P \ {α}

    for every β in T where U_T[β] < u_t:
        P ← P \ {β}    ← "covered" — no need to ever label
```

### The Kernel Uncertainty U_T

```
U_T[i] = K(x_i, x_i)  −  K(x_i, S) · K(S,S)⁻¹ · K(S, x_i)
         ↑ prior var       ↑ how much S already "explains" x_i
```

- Composition **similar** to many in S → `K(x_i, S)` large → subtracted term large → `U_T[i]` near zero → skip
- Composition **far** from everything in S → `K(x_i, S)` small → `U_T[i]` ≈ prior variance → add it

**Cholesky implementation for numerical stability:**
```python
K_TT_diag = kernel.diag(X_T)
K_TS      = kernel(X_T, X_S)
K_SS_reg  = kernel(X_S, X_S) + 1e-8 * I
L         = cholesky(K_SS_reg)
V         = L⁻¹ K_TS.T                         # (|S|, |T|)
U         = K_TT_diag - einsum("ij,ij->j", V, V)
```

### The Two Pruning Decisions Per Iteration

| Action | Condition | Effect |
|--------|-----------|--------|
| **Add** `argmax(U_T)` to S | `U[best] > u_t = 1e-6` | Grows selected set |
| **Remove** all β from P | `U[β] < u_t` | Permanently discard "covered" points |

The second rule is the speedup: once S covers a region, all nearby pool points are pruned from P forever. P shrinks fast and the loop terminates well before exhausting all 1000 compositions.

### Settings Used

```python
N_t         = None    # evaluate ALL of P each step (no subsampling)
U_THRESHOLD = 1e-6    # very small → nearly greedy (always add the max)
N_MAX       = 100
seed        = 770077
```

### Snapshots for the Learning Curve

```python
al_selections[N] = al_order[:N]   # nested — N=20 is a subset of N=30
```

P_bubble is computed by PC-SAFT **after** selection — the entire selection loop is label-free.

---

## 4. Active Learning — Label-Free RBF (`04_AL_Sampling.ipynb`)

### Philosophy Shift

`02_GPR_AL` needs the A2 labeled dataset to calibrate the kernel. `04_AL_Sampling` is **fully label-free from the start** — it estimates everything it needs from the pool geometry.

### Feature Representation

```python
z_i   = x_i / (1 - x_CO2)                          # impurity species fraction
X_raw = [x_CO2,  z_H2, z_Ar, z_N2, z_CH4, z_O2, z_CO, z_H2S]
X_scaled = MinMaxScaler().fit_transform(X_raw)
```

CO2 level and impurity *pattern* are now independent axes. Same normalization as the stratified sampling.

### Length Scale — Estimated from Pool Geometry

```python
nn_dists = NearestNeighbors(n_neighbors=2).fit(X_scaled)
l_nn     = median(nn_dists)          # ≈ 0.145  (median NN gap in scaled space)
l_scale  = 5 × l_nn                 # = 0.726
```

The `5×` multiplier is chosen from a diagnostic plot: too small (`1×`) → flat curve (kernel sees only immediate neighbours); too large (`10×`) → loop terminates too fast. `5×` gives smooth monotonic decay.

### The Greedy Loop (no pruning)

Uses an isotropic RBF kernel — no per-feature length-scales:
```
k(a, b) = σ_f² · exp( −||a−b||² / (2 l²) )
```

Every step picks the point with the highest posterior variance given S. No convergence threshold — all 98 steps run unconditionally.

### Uncertainty Decay

```
N= 20  →  σ² = 0.268
N= 40  →  σ² = 0.137
N= 60  →  σ² = 0.038
N= 80  →  σ² = 0.018
N=100  →  σ² = 0.013
```

### Comparison: `02_GPR_AL` vs `04_AL_Sampling`

| | `02_GPR_AL` | `04_AL_Sampling` |
|---|---|---|
| Kernel | Matérn-5/2 ARD | Isotropic RBF |
| Length-scales | Fitted on A2 labeled data | Median NN distance × 5 |
| Feature space | Raw z + StandardScaler | CO2 + normalized impurities + MinMaxScaler |
| Requires labels? | Yes (A2 for kernel fitting) | No — fully label-free |
| Pruning (Algorithm 1)? | Yes — removes covered P points | No — all steps run |
| Per-component relevance? | Yes — ARD captures N2 > H2S | No — all dims treated equally |

---

## 5. How the GPR Kernel Hyperparameters Are Fitted

### What θ Is

All learnable numbers in one vector:
```
θ = [ σ_f²,   l_CO2, l_H2, l_Ar, l_N2, l_CH4, l_O2, l_CO, l_H2S,   σ_n² ]
```
That is 10 numbers. The optimizer moves all 10 simultaneously.

### Step 1 — Build K from θ

For each pair of training compositions (i, j), compute the ARD distance:
```
r_ij  =  sqrt(  (x_i,CO2 - x_j,CO2)² / l_CO2²
              + (x_i,N2  - x_j,N2 )² / l_N2²
              + ...  )
```

Then the kernel entry:
```
K_θ[i,j]  =  σ_f²  ×  Matérn_5/2(r_ij)
           =  σ_f²  ×  (1 + √5·r + 5r²/3) · exp(-√5·r)
```

Add noise on the diagonal: `K_full = K_θ + σ_n² · I`

This gives a 5000×5000 matrix where each entry encodes how similar two compositions are **in the directions θ currently considers important**.

### Step 2 — Compute the Log Marginal Likelihood

```
log p(y | X, θ)  =  - ½ yᵀ K_full⁻¹ y       ← data fit
                   - ½ log det(K_full)         ← complexity penalty
                   - n/2 log(2π)              ← constant
```

**Data fit term:** Solve `K_full · α = y` via Cholesky, compute `-½ yᵀα`. Large when similar compositions (per current θ) genuinely have similar P_bubble.

**Complexity penalty:** Penalises an overly flexible kernel. If length-scales are tiny, every point looks unique, the kernel fits perfectly but learns nothing. The two terms compete → the optimizer finds the sweet spot.

### Step 3 — Compute the Gradient ∂log p / ∂θ

Closed-form analytic gradient:
```
∂ log p / ∂θ_j  =  ½ tr( (α αᵀ  −  K_full⁻¹)  ·  ∂K_full/∂θ_j )
```

For length-scale `l_k`:
```
∂K_θ[i,j] / ∂l_k  =  σ_f²  ×  d(Matérn)/dr  ×  ∂r_ij/∂l_k

∂r_ij/∂l_k  =  - (x_i,k - x_j,k)² / (l_k³ · r_ij)
```

If compositions i and j differ a lot in N2, then `∂K[i,j]/∂l_N2` is large — the gradient tells the optimizer: "shrinking l_N2 changes the fit a lot, N2 matters."

### Step 4 — L-BFGS-B Update

```
1. Evaluate log p(θ_current) and ∇_θ log p
2. Approximate Hessian H from last ~10 gradient history vectors
3. Search direction: d = -H⁻¹ ∇ log p   (gradient ascent scaled by curvature)
4. Line search along d for step size
5. θ_new = θ_current + step_size × d
6. Repeat until ||∇ log p|| < tolerance
```

Internally sklearn works in **log space** (`log l_i` instead of `l_i`) so length-scales can never go negative. Bounds `(1e-3, 1e3)` are enforced at every step.

`n_restarts_optimizer=3` runs the entire process 4 times from different random `θ₀`; the best converged result is kept.

### Full Optimization Trace (schematic)

```
θ₀ = [σ_f²=1.0,  l_i=1.0 for all i,  σ_n²=0.1]

Iter 1:  Build K(θ₀) → 5000×5000
         Solve K·α = y (Cholesky)
         log p = -47823.1
         ∇_θ:  l_N2 gradient large  → shrink l_N2
                l_H2S gradient small → leave l_H2S alone
         L-BFGS-B step → θ₁

Iter 2:  log p = -47301.6  ← better
         ...

Iter ~50: ||∇ log p|| < tol → converge
θ* = [σ_f²=0.8,  l_CO2=0.4, l_N2=0.12, ..., l_H2S=2.3,  σ_n²=0.05]
```

---

## 6. Physical Meaning of the ARD Length-Scales

The length-scale for component i answers: **"how much do I need to change x_i before P_bubble noticeably changes?"**

| Component | Tc (K) | Status at T=250 K | Effect on P_bubble | Expected l_i |
|---|---|---|---|---|
| N2 | 126 | Supercritical | Strong — raises P_bubble sharply | Short |
| CH4 | 190 | Supercritical | Strong | Short |
| H2 | 33 | Far supercritical | Large per-mole effect | Short–moderate |
| Ar | 151 | Supercritical | Moderate | Moderate |
| CO | 132 | Supercritical | Moderate | Moderate |
| O2 | 154 | Supercritical | Moderate | Moderate |
| H2S | 373 | **Subcritical** | Condensable — soft, complex effect | Long |
| CO2 | 304 | Subcritical | Dominant component — very sensitive | Short |

Light gases (N2, CH4, H2) are far above their critical temperatures at 250 K — non-condensable. Adding even a small amount forces the bubble point pressure up significantly. The GPR sees this variance in A2 and assigns them **short length-scales**.

H2S is subcritical at 250 K — condensable — and tends to absorb into the CO2-rich liquid rather than drive the bubble point up. Assigned a **long length-scale**: "two compositions differing only in H2S look similar for P_bubble purposes."

**The optimizer never reads a chemistry textbook.** It infers the equivalent information from the empirical covariance structure of ~5000 PC-SAFT-labeled compositions. The physics enters through the data.

### Consequence for AL Selection

When `compute_U_T()` uses the fitted kernel, "how covered is this composition?" is measured in units weighted by thermodynamic importance:

- New composition with very different N2 or CH4 from anything in S → large U_T → gets selected
- New composition that only differs in H2S → small U_T → looks like something already in S → skipped

This is what `04_AL_Sampling` (isotropic RBF) cannot do — it treats a 0.1 change in z_N2 and a 0.1 change in z_H2S identically after scaling.
