import time
notebook_start = time.perf_counter()
# %pip install -e /home/darshan/A6/PCSAFT_cDFT/thermoift

import os
import numpy  as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import thermoift.PLOT_SETTINGS as ps
from thermoift.rng_utils import get_rng
from thermoift import print_model_metrics
from thermoift.FeosPlugin import (
    RegistryManager, ParameterBuilder, VLECalculator, CompositionHandler,
)
import feos
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

RegistryManager.load_registry()


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

POOL_PATH = "AL_POOL/composition_pool.csv"
A2_PATH   = "../DATASET_A2/CombinedDataset_A2.csv"

T_REF = 250.0   # K — reference temperature for GPR hyperparameter fitting

# Sample sizes for learning-curve comparison
N_SIZES = list(range(10, 110, 10))   # [10, 20, 30, …, 100]
N_MAX   = max(N_SIZES)               # 100

# Algorithm 1 parameters
N_t         = None   # pool-evaluation budget per AL step (None → full P)
U_THRESHOLD = 1e-6   # u_t: kernel-uncertainty cut-off (small → pure greedy)

# GPR fitting on A2 data
GPR_MAX_SAMPLES   = 5000
RESTART_OPTIMIZER = 3
SEED              = 770077

OUTPUT_DIR = "AL_OUTPUTS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"N_SIZES  : {N_SIZES}")
print(f"T_REF    : {T_REF} K")
print(f"N_t      : {N_t}   (None = full P each step)")
print(f"u_t      : {U_THRESHOLD}")


# ── Column-name bridge ────────────────────────────────────────────────────────
CSV_TO_FULL = {
    "CO2" : "carbon dioxide",
    "H2"  : "hydrogen",
    "Ar"  : "argon",
    "N2"  : "nitrogen",
    "CH4" : "methane",
    "O2"  : "oxygen",
    "CO"  : "carbon monoxide",
    "H2S" : "hydrogen sulfide",
}
POOL_Z_COLS = list(CSV_TO_FULL.keys())            # short names in pool CSV
Z_FEATURES  = [f"z_{v}" for v in CSV_TO_FULL.values()]  # long names in A2

COMP_MAP = {k: f"z_{v}" for k, v in CSV_TO_FULL.items()}  # CO2→z_carbon dioxide…

FEATURES    = Z_FEATURES
TARGET      = "P_bubble"
TARGET_UNIT = "bar"

print(f"Pool  columns : {POOL_Z_COLS}")
print(f"GPR features  : {FEATURES}")


# ── Load candidate pool (compositions only, 1000 rows) ────────────────────────
pool = pd.read_csv(POOL_PATH)
print(f"Pool loaded: {len(pool):,} compositions")
print(f"Columns: {list(pool.columns)}")
print()
print(pool.head(3))


# ── Load A2, filter to 8-component rows at T_REF, fit GPR ─────────────────────
a2 = pd.read_csv(A2_PATH)
a2.columns = [c.strip() for c in a2.columns]

# Keep only rows where extra components (water, SO2, propane, ethane, etc.)
# contribute nothing — ensures z-vector sums to 1 for our 8 species
extra_z = [c for c in a2.columns if c.startswith("z_") and c not in Z_FEATURES]
if extra_z:
    a2 = a2[a2[extra_z].sum(axis=1) < 1e-6].copy()
    print(f"8-component rows: {len(a2):,}")

# Prefer rows at T_REF ± 1 K; fall back to all temperatures if too few
a2_ref = a2[np.abs(a2["temperature"] - T_REF) <= 1.0].copy()
if len(a2_ref) < 100:
    print(f"Only {len(a2_ref)} rows at T_REF — using all temperatures")
    a2_ref = a2.copy()

# Use pre-computed P_bubble from A2 (no need to re-run PC-SAFT for training)
df_train = (a2_ref[Z_FEATURES + [TARGET]]
            .drop_duplicates(subset=Z_FEATURES)
            .dropna(subset=[TARGET])
            .reset_index(drop=True))

if len(df_train) > GPR_MAX_SAMPLES:
    df_train = df_train.sample(n=GPR_MAX_SAMPLES, random_state=SEED).reset_index(drop=True)

print(f"GPR training set : {len(df_train):,} labeled A2 compositions")
print(f"P_bubble range   : {df_train[TARGET].min():.2f} – {df_train[TARGET].max():.2f} bar")

# ── Fit GPR: z → P_bubble(T_REF) ──────────────────────────────────────────────
n_feat = len(FEATURES)
kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * Matern(nu=2.5, length_scale=np.ones(n_feat), length_scale_bounds=(1e-3, 1e3))
    + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 100.0))
)

gpr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("gpr",    GaussianProcessRegressor(
        kernel               = kernel,
        alpha                = 0.0,
        normalize_y          = True,
        n_restarts_optimizer = RESTART_OPTIMIZER,
        random_state         = SEED,
    )),
])

t0 = time.perf_counter()
gpr_model.fit(df_train[FEATURES], df_train[TARGET])
print(f"\nGPR fitting time : {time.perf_counter()-t0:.1f} s")


# ── GPR diagnostics and ARD feature importances ───────────────────────────────
gpr_step = gpr_model.named_steps["gpr"]
fitted_k = gpr_step.kernel_   # full fitted kernel

# Extract signal kernel (ConstantKernel × Matern) — used for AL uncertainty
# kernel_ structure: (ConstantKernel * Matern) + WhiteKernel → k1 + k2
signal_kernel = fitted_k.k1   # ConstantKernel × Matern  (no noise)

print("Fitted kernel:"); print(f"  {fitted_k}")
print()

# ARD length-scales → feature importances
ls   = signal_kernel.k2.length_scale          # one per feature
imp  = 1.0 / ls;  imp /= imp.sum()
print("ARD feature importances (1 / length-scale, normalised):")
for name, val in sorted(zip(FEATURES, imp), key=lambda x: -x[1]):
    bar = "█" * max(1, int(val * 40))
    print(f"  {name:<30s}  {val*100:5.1f}%  {bar}")

# Training-set fit quality
y_pred, _ = gpr_model.predict(df_train[FEATURES], return_std=True)
res = df_train[TARGET].values - y_pred
print(f"\nTrain RMSE : {np.sqrt(np.mean(res**2)):.3f} bar")
print(f"Train R²   : {1 - np.var(res)/np.var(df_train[TARGET].values):.4f}")


# ── Scale pool features using fitted StandardScaler ───────────────────────────
# The GPR kernel was optimised on scaled features → apply same transform.
scaler = gpr_model.named_steps["scaler"]

pool_renamed = pool.rename(columns=COMP_MAP)   # CO2 → z_carbon dioxide …
X_pool_raw   = pool_renamed[FEATURES].values   # (1000, 8)  unscaled
X_pool       = scaler.transform(X_pool_raw)    # (1000, 8)  scaled

print(f"Pool feature matrix shape : {X_pool.shape}")
print(f"Max kernel variance k(x,x): {signal_kernel.diag(X_pool[:5]).mean():.4f}  (should be ≈ C × 1)")


# ── Algorithm 1 — kernel-based active learning (Xiang et al. 2023) ────────────

def compute_U_T(X_T, X_S, kernel):
    """
    Kernel uncertainty: U_T = diag(K_TT - K_TS K_SS⁻¹ K_ST)
    X_T : (|T|, d) — candidate batch
    X_S : (|S|, d) — already-selected set
    """
    K_TT_diag = kernel.diag(X_T)                      # (|T|,)
    K_TS      = kernel(X_T, X_S)                      # (|T|, |S|)
    K_SS      = kernel(X_S, X_S)                      # (|S|, |S|)
    K_SS_reg  = K_SS + 1e-8 * np.eye(len(X_S))
    try:
        L = np.linalg.cholesky(K_SS_reg)
        V = np.linalg.solve(L, K_TS.T)               # (|S|, |T|)
        U = K_TT_diag - np.einsum("ij,ij->j", V, V)
    except np.linalg.LinAlgError:
        K_inv = np.linalg.pinv(K_SS_reg)
        U = K_TT_diag - np.einsum("ij,jk,ki->i", K_TS, K_inv, K_TS.T)
    return np.maximum(U, 0.0)


def run_algorithm1(X_all, kernel, n_max, n_t=None, u_t=1e-6, seed=0):
    """
    Returns a list of pool indices in the order they were added to S.
    Starts with 2 random seeds; grows until |S| = n_max or P is empty.
    """
    rng     = np.random.default_rng(seed)
    n_pool  = len(X_all)
    idx_arr = np.arange(n_pool)

    # Initial S: 2 random points
    init    = rng.choice(n_pool, size=2, replace=False)
    s_list  = list(init)                 # ordered selection history
    p_set   = set(idx_arr) - set(s_list) # remaining candidates

    while p_set and len(s_list) < n_max:
        p_list = list(p_set)

        # Optionally subsample T from P
        if n_t is not None and len(p_list) > n_t:
            t_idx = rng.choice(len(p_list), size=n_t, replace=False)
            t_list = [p_list[i] for i in t_idx]
        else:
            t_list = p_list

        X_T = X_all[t_list]              # (|T|, d)
        X_S = X_all[s_list]              # (|S|, d)

        U = compute_U_T(X_T, X_S, kernel)  # (|T|,)

        # Add argmax to S if above threshold
        best_local = int(np.argmax(U))
        if U[best_local] > u_t:
            best_global = t_list[best_local]
            s_list.append(best_global)
            p_set.discard(best_global)

        # Remove "converged" points (U < u_t) from P → C (discard)
        for i, idx in enumerate(t_list):
            if U[i] < u_t:
                p_set.discard(idx)

    return s_list


print("Running Algorithm 1 …")
t0 = time.perf_counter()
al_order = run_algorithm1(
    X_all  = X_pool,
    kernel = signal_kernel,
    n_max  = N_MAX,
    n_t    = N_t,
    u_t    = U_THRESHOLD,
    seed   = SEED,
)
elapsed_al = time.perf_counter() - t0
print(f"Algorithm 1 done: {len(al_order)} points selected in {elapsed_al:.1f} s")

# Build AL snapshots
al_selections = {}
for N in N_SIZES:
    n_avail = min(N, len(al_order))
    chosen  = al_order[:n_avail]
    if n_avail < N:
        # Fill shortage with random picks from the rest of the pool
        fallback = [i for i in range(len(pool)) if i not in set(chosen)]
        rng_fb   = np.random.default_rng(SEED + N)
        rng_fb.shuffle(fallback)
        chosen = chosen + fallback[: N - n_avail]
    al_selections[N] = chosen

print("\nAL snapshot sizes:")
for N in N_SIZES:
    print(f"  N={N:3d}  selected={len(al_selections[N])}")


# ── Random and Stratified selections from the same pool ───────────────────────
rng_base = np.random.default_rng(SEED)
perm     = rng_base.permutation(len(pool))   # one shuffle; subsets are nested

random_selections = {N: list(perm[:N]) for N in N_SIZES}


def stratified_select(n, df, seed):
    """
    Proportional CO2-bin × mixture_size stratified sample of size n.
    """
    sp = df.copy()
    sp["_co2_bin"] = pd.cut(
        sp["CO2"],
        bins=[0.0, 0.94, 0.96, 0.97, 0.98, 0.99, 1.01],
        labels=["<0.94", "0.94-0.96", "0.96-0.97", "0.97-0.98", "0.98-0.99", "≥0.99"],
    )
    counts = sp.groupby(["_co2_bin", "mixture_size"], observed=True).size()
    alloc  = (counts / counts.sum() * n).round().clip(lower=1).astype(int)
    # Adjust total to exactly n
    while alloc.sum() > n: alloc[alloc.idxmax()] -= 1
    while alloc.sum() < n: alloc[alloc.idxmin()] += 1
    frames = []
    for (cb, ms), k in alloc.items():
        sub = sp[(sp["_co2_bin"] == cb) & (sp["mixture_size"] == ms)]
        if len(sub):
            frames.append(sub.sample(n=min(k, len(sub)), random_state=seed))
    result = (pd.concat(frames)
                .drop(columns=["_co2_bin"])
                .drop_duplicates()
                .head(n))
    return list(result.index)

stratified_selections = {N: stratified_select(N, pool, SEED) for N in N_SIZES}

print("Random selections (nested shuffle):")
for N in N_SIZES:
    print(f"  N={N:3d}  count={len(random_selections[N])}")
print("\nStratified selections:")
for N in N_SIZES:
    print(f"  N={N:3d}  count={len(stratified_selections[N])}")


# ── Compute P_bubble(T_REF) for every selected composition via PC-SAFT ─────────
# Collect all unique pool indices across all selections and strategies.
ALL_COMPS_FULL = [CSV_TO_FULL[k] for k in POOL_Z_COLS]

def compute_pbubble_single(z_arr, T_K):
    """PC-SAFT bubble-point. Returns P in bar, or NaN on failure."""
    active_z, active_comps, _ = CompositionHandler.reduce_components(
        z_arr, ALL_COMPS_FULL, verbose=False)
    params = ParameterBuilder.build_parameters(active_comps)
    func   = feos.HelmholtzEnergyFunctional.pcsaft(params)
    feed   = CompositionHandler.compute_feed_moles(active_z)
    T_bub, P_bub = VLECalculator.compute_bubble_curve(
        func, [T_K], feed, verbose=False)
    return float(P_bub[0]) if len(P_bub) else float("nan")


needed_idx = set()
for sel in al_selections.values():     needed_idx.update(sel)
for sel in random_selections.values(): needed_idx.update(sel)
for sel in stratified_selections.values(): needed_idx.update(sel)
needed_idx = sorted(needed_idx)
print(f"Unique pool compositions to evaluate: {len(needed_idx)}")

pbubble_cache = {}   # pool_idx → P_bubble (bar)
n_failed = 0
t0 = time.perf_counter()

for i, idx in enumerate(needed_idx):
    row   = pool.iloc[idx]
    z_arr = np.array([row[k] for k in POOL_Z_COLS])
    try:
        p = compute_pbubble_single(z_arr, T_REF)
    except Exception as e:
        if n_failed < 3:
            print(f"  [warn] pool[{idx}]: {e}")
        p = float("nan")
        n_failed += 1
    pbubble_cache[idx] = p
    if (i + 1) % 20 == 0:
        print(f"  {i+1:4d}/{len(needed_idx)}  ({n_failed} failed so far) …")

elapsed_pb = time.perf_counter() - t0
n_ok = sum(1 for v in pbubble_cache.values() if not np.isnan(v))
print(f"\nP_bubble computed: {n_ok}/{len(needed_idx)} OK, {n_failed} failed")
print(f"Time: {elapsed_pb:.1f} s  ({elapsed_pb/max(len(needed_idx),1):.2f} s/composition)")


# ── Assemble selection DataFrames (pool row + P_bubble column) ─────────────────

def make_selection_df(idx_list, pool_df, pbubble_cache):
    rows = pool_df.iloc[idx_list].copy().reset_index(drop=True)
    rows["pool_idx"] = idx_list
    rows["P_bubble"] = [pbubble_cache.get(i, float("nan")) for i in idx_list]
    return rows

selections_df = {}
for N in N_SIZES:
    selections_df[("AL",         N)] = make_selection_df(al_selections[N],         pool, pbubble_cache)
    selections_df[("Random",     N)] = make_selection_df(random_selections[N],     pool, pbubble_cache)
    selections_df[("Stratified", N)] = make_selection_df(stratified_selections[N], pool, pbubble_cache)

print("Selection DataFrames built.  Summary (P_bubble OK count):")
for (strat, N), df in selections_df.items():
    n_ok = df["P_bubble"].notna().sum()
    print(f"  {strat:12s}  N={N:3d}   P_bubble_ok={n_ok}/{N}")


# ── Learning-curve plots ──────────────────────────────────────────────────────
strategy_styles = {
    "AL":         ("crimson",    "-",  "o"),
    "Stratified": ("darkorange", "--", "s"),
    "Random":     ("royalblue",  ":",  "^"),
}

# Compute per-strategy mean P_bubble vs N
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
for strat, (color, ls, mk) in strategy_styles.items():
    means = [selections_df[(strat, N)]["P_bubble"].mean() for N in N_SIZES]
    ax.plot(N_SIZES, means, color=color, linestyle=ls, linewidth=2,
            marker=mk, markersize=5, label=strat)
ax.set_xlabel("Sample budget N")
ax.set_ylabel(f"Mean P_bubble at {T_REF} K (bar)")
ax.set_title("Mean bubble-point pressure vs sample size")
ax.legend(fontsize=9)
ps.apply_axis_style(ax)

# CO2 fraction coverage
ax2 = axes[1]
for strat, (color, ls, mk) in strategy_styles.items():
    co2_stds = [selections_df[(strat, N)]["CO2"].std() for N in N_SIZES]
    ax2.plot(N_SIZES, co2_stds, color=color, linestyle=ls, linewidth=2,
             marker=mk, markersize=5, label=strat)
ax2.set_xlabel("Sample budget N")
ax2.set_ylabel("Std(CO2 fraction)")
ax2.set_title("CO2 fraction diversity vs sample size")
ax2.legend(fontsize=9)
ps.apply_axis_style(ax2)

plt.tight_layout()
ps.save_plot(fig, "AL_comparison_curves", folder=OUTPUT_DIR)
plt.show()

# ── AL selection order in CO2 space (N=100) ───────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(8, 4))
df_al100 = selections_df[("AL", N_MAX)]
order_in_pool = df_al100["pool_idx"].values
ax3.scatter(range(len(order_in_pool)), df_al100["CO2"].values,
            c=range(len(order_in_pool)), cmap="plasma", s=20, alpha=0.8)
ax3.set_xlabel("Selection order"); ax3.set_ylabel("CO2 fraction")
ax3.set_title(f"AL selection order (N={N_MAX}) in CO2 space")
plt.colorbar(ax3.collections[0], ax=ax3, label="Selection step")
ps.apply_axis_style(ax3)
plt.tight_layout()
ps.save_plot(fig2, "AL_selection_order_CO2", folder=OUTPUT_DIR)
plt.show()


# ── Save all 30 CSV files: one per (strategy, N) ─────────────────────────────
STRAT_DIR = {
    "AL":         os.path.join(OUTPUT_DIR, "AL"),
    "Random":     os.path.join(OUTPUT_DIR, "Random"),
    "Stratified": os.path.join(OUTPUT_DIR, "Stratified"),
}
for d in STRAT_DIR.values():
    os.makedirs(d, exist_ok=True)

for (strat, N), df in selections_df.items():
    fname = os.path.join(STRAT_DIR[strat], f"{strat}_N{N:03d}.csv")
    df.to_csv(fname, index=False)

print("Saved:")
for (strat, N) in sorted(selections_df.keys()):
    fname = os.path.join(STRAT_DIR[strat], f"{strat}_N{N:03d}.csv")
    print(f"  {fname}")

# ── Also save the fitted GPR model ───────────────────────────────────────────
model_path = os.path.join(OUTPUT_DIR, "GPR_AL_model.joblib")
joblib.dump(gpr_model, model_path)
print(f"\nGPR model saved → {model_path}")


elapsed_total = (time.perf_counter() - notebook_start) / 60
print(f"Total notebook runtime : {elapsed_total:.1f} min")
print(f"  Algorithm 1 (kernel) : {elapsed_al:.1f} s")
print(f"  P_bubble (PC-SAFT)   : {elapsed_pb:.1f} s")
