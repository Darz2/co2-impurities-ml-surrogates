"""
04_AL_boltzmann.py
──────────────────
Experiment: replace argmax(U) with Boltzmann-weighted sampling
    P(i) ∝ exp(U_i / T)
to interpolate between
    T → 0    : pure greedy MaxVar (current AL behaviour)
    T → ∞    : uniform random
and trace the inverted-U exploration/exploitation curve on the
composition pool.

What this script DOES:
  - reuses the same GPR fit and signal kernel as 02_GPR_AL.py
  - runs greedy AL (T=0), Boltzmann at T ∈ {0.1, 1, 10}·σ_f²*, and
    a uniform-random baseline (T = ∞) using the SAME loop structure
  - visualises selection ordering, pool coverage in PCA space,
    and pairwise-distance histograms

It does NOT call PC-SAFT — we focus on the geometry of selection,
not the labels. That keeps the script fast (< 1 min).

Run from this directory:
    python 04_AL_boltzmann.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing      import StandardScaler
from sklearn.pipeline           import Pipeline
from sklearn.gaussian_process   import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.decomposition      import PCA

import thermoift.PLOT_SETTINGS as ps

t0_all = time.perf_counter()

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
PART2     = HERE.parent.parent
POOL_PATH = PART2 / "POOL" / "composition_pool.csv"
A2_PATH   = PART2.parent / "DATASET_A2" / "CombinedDataset_A2.csv"
OUT_DIR   = HERE / "OUT_boltzmann"
OUT_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
T_REF     = 250.0
N_MAX     = 100
SEED      = 770077
# Floor is set very small relative to σ_f²* so pruning effectively never
# fires — we want every strategy to reach the full budget N_MAX so that
# their selection trajectories are directly comparable.
U_T_FLOOR = 1e-15
GPR_MAX   = 5000
N_RESTART = 3

# Temperatures expressed as fractions of σ_f²* (signal variance).
# 0  → pure greedy ; inf → pure random ; intermediate values trace the curve.
T_FRACS = [0.0, 0.1, 1.0, 10.0, np.inf]

CSV_TO_FULL = {"CO2":"carbon dioxide","H2":"hydrogen","Ar":"argon","N2":"nitrogen",
               "CH4":"methane","O2":"oxygen","CO":"carbon monoxide","H2S":"hydrogen sulfide"}
POOL_Z_COLS = list(CSV_TO_FULL.keys())
Z_FEATURES  = [f"z_{v}" for v in CSV_TO_FULL.values()]
COMP_MAP    = {k: f"z_{v}" for k, v in CSV_TO_FULL.items()}
TARGET      = "P_bubble"


# ── Load data and fit GPR ─────────────────────────────────────────────────────
print("Loading data …")
pool = pd.read_csv(POOL_PATH)
a2   = pd.read_csv(A2_PATH)
a2.columns = [c.strip() for c in a2.columns]
extra_z = [c for c in a2.columns if c.startswith("z_") and c not in Z_FEATURES]
if extra_z:
    a2 = a2[a2[extra_z].sum(axis=1) < 1e-6].copy()
a2_ref = a2[np.abs(a2["temperature"] - T_REF) <= 1.0].copy()
if len(a2_ref) < 100:
    a2_ref = a2.copy()
df_train = (a2_ref[Z_FEATURES + [TARGET]]
              .drop_duplicates(subset=Z_FEATURES)
              .dropna(subset=[TARGET])
              .reset_index(drop=True))
if len(df_train) > GPR_MAX:
    df_train = df_train.sample(n=GPR_MAX, random_state=SEED).reset_index(drop=True)

print(f"Fitting GPR on {len(df_train):,} rows …")
n_feat = len(Z_FEATURES)
kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * Matern(nu=2.5, length_scale=np.ones(n_feat), length_scale_bounds=(1e-3, 1e3))
    + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 100.0))
)
gpr = Pipeline([
    ("scaler", StandardScaler()),
    ("gpr",    GaussianProcessRegressor(
        kernel=kernel, alpha=0.0, normalize_y=True,
        n_restarts_optimizer=N_RESTART, random_state=SEED)),
])
t0 = time.perf_counter()
gpr.fit(df_train[Z_FEATURES], df_train[TARGET])
print(f"  fit time: {time.perf_counter()-t0:.1f} s")

scaler        = gpr.named_steps["scaler"]
signal_kernel = gpr.named_steps["gpr"].kernel_.k1
sigma_f_sq    = float(signal_kernel.k1.constant_value)
print(f"  σ_f²*   = {sigma_f_sq:.4f}")

X_pool = scaler.transform(pool.rename(columns=COMP_MAP)[Z_FEATURES].values)


# ── Selection routines --------------------------------------------------------
def compute_U(X_T, X_S, kern):
    K_TT_diag = kern.diag(X_T)
    K_TS      = kern(X_T, X_S)
    K_SS      = kern(X_S, X_S) + 1e-8 * np.eye(len(X_S))
    L = np.linalg.cholesky(K_SS)
    V = np.linalg.solve(L, K_TS.T)
    return np.maximum(K_TT_diag - np.einsum("ij,ij->j", V, V), 0.0)


def boltzmann_pick(U, T, rng):
    """Return an index into U sampled with prob ∝ exp(U/T)."""
    if T <= 0.0:
        return int(np.argmax(U))
    if not np.isfinite(T):
        return int(rng.integers(len(U)))
    logits = U / T
    logits -= logits.max()                         # softmax stability
    p = np.exp(logits)
    p_sum = p.sum()
    if not np.isfinite(p_sum) or p_sum == 0.0:
        return int(np.argmax(U))
    return int(rng.choice(len(U), p=p / p_sum))


def run_al_boltzmann(X_all, kern, n_max, T, seed, u_t=U_T_FLOOR):
    rng    = np.random.default_rng(seed)
    n_pool = len(X_all)
    s_list = list(rng.choice(n_pool, size=2, replace=False))
    p_set  = set(range(n_pool)) - set(s_list)
    u_pick = [np.nan, np.nan]

    while p_set and len(s_list) < n_max:
        p_list = list(p_set)
        U      = compute_U(X_all[p_list], X_all[s_list], kern)
        idx    = boltzmann_pick(U, T, rng)
        if U[idx] <= u_t and T <= 0.0:
            break
        s_list.append(p_list[idx])
        u_pick.append(float(U[idx]))
        p_set.discard(p_list[idx])
        for i, gidx in enumerate(p_list):
            if U[i] < u_t:
                p_set.discard(gidx)
    return s_list, u_pick


# ── Run all temperatures ------------------------------------------------------
print("\nRunning Boltzmann sweep …")
results = {}
for f in T_FRACS:
    T = (f * sigma_f_sq) if np.isfinite(f) else np.inf
    t0 = time.perf_counter()
    s, u_p = run_al_boltzmann(X_pool, signal_kernel, N_MAX, T, SEED)
    label = (r"Greedy ($T\!\to\!0$)"      if f == 0.0
             else r"Random ($T\!\to\!\infty$)" if not np.isfinite(f)
             else rf"$T = {f:g}\,\sigma_f^{{2*}}$")
    results[f] = {"T": T, "s": s, "u_pick": u_p, "label": label,
                  "time": time.perf_counter() - t0}
    print(f"  {label:30s}  |S|={len(s):3d}   t={results[f]['time']:5.2f}s")


# ── PCA of the pool (used for spatial plots) ---------------------------------
pca_pool = PCA(n_components=2, random_state=SEED).fit(X_pool)
pool_2d  = pca_pool.transform(X_pool)
print(f"\nPCA explained variance: {pca_pool.explained_variance_ratio_.sum():.2%}")


# ── Plot 1: pool coverage in PCA space for each temperature ------------------
fig, axes = plt.subplots(1, len(T_FRACS), figsize=(3.4 * len(T_FRACS), 3.6),
                         sharex=True, sharey=True)
for ax, f in zip(axes, T_FRACS):
    ax.scatter(pool_2d[:, 0], pool_2d[:, 1], s=8, color="0.85",
               edgecolors="none", label="pool" if f == T_FRACS[0] else None)
    s = results[f]["s"]
    sel = pool_2d[s]
    sc = ax.scatter(sel[:, 0], sel[:, 1],
                    c=np.arange(len(sel)), cmap="plasma",
                    s=24, edgecolors="k", linewidths=0.4)
    ax.set_title(results[f]["label"], fontsize=10)
    ax.set_xlabel("PC 1")
    ps.apply_axis_style(ax)
axes[0].set_ylabel("PC 2")
cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.01)
cbar.set_label("selection order")
ps.save_plot(fig, "01_pca_coverage", folder=str(OUT_DIR))
plt.close(fig)


# ── Plot 2: CO2-fraction order along selection ------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(T_FRACS)))
for col, f in zip(colors, T_FRACS):
    co2 = pool.iloc[results[f]["s"]]["CO2"].values
    ax.plot(range(1, len(co2) + 1), co2, "-", color=col, lw=1.2, alpha=0.85,
            label=results[f]["label"])
ax.set_xlabel("selection step")
ax.set_ylabel(r"CO$_2$ mole fraction of selected composition")
ax.set_title(r"How each strategy traverses CO$_2$ space")
ax.legend(fontsize=8, loc="best")
ps.apply_axis_style(ax)
plt.tight_layout()
ps.save_plot(fig, "02_CO2_order", folder=str(OUT_DIR))
plt.close(fig)


# ── Plot 3: nearest-neighbour distance histogram inside S --------------------
def nn_distances(idxs):
    """Min pairwise Euclidean distance in the standardised feature space."""
    pts = X_pool[idxs]
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1)

fig, ax = plt.subplots(figsize=(8, 4.5))
for col, f in zip(colors, T_FRACS):
    d = nn_distances(results[f]["s"])
    ax.hist(d, bins=20, histtype="step", lw=2, color=col,
            label=fr"{results[f]['label']}   median NN = {np.median(d):.2f}")
ax.set_xlabel(r"nearest-neighbour distance in $\tilde{\mathbf{x}}$ (ARD-scaled)")
ax.set_ylabel("count in selected set")
ax.set_title("Spread of the selected set: greedy = sparse, random = uneven")
ax.legend(fontsize=8)
ps.apply_axis_style(ax)
plt.tight_layout()
ps.save_plot(fig, "03_nn_distance", folder=str(OUT_DIR))
plt.close(fig)


# ── Plot 4: kernel-uncertainty at moment of pick -----------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
for col, f in zip(colors, T_FRACS):
    u_p = results[f]["u_pick"]
    ax.plot(range(1, len(u_p) + 1), u_p, "-", lw=1.4, color=col,
            label=results[f]["label"])
ax.axhline(sigma_f_sq, ls="--", color="0.4", lw=1,
           label=fr"max  $\sigma_f^{{2*}} = {sigma_f_sq:.3f}$")
ax.set_yscale("log")
ax.set_xlabel("selection step")
ax.set_ylabel(r"$U(\mathbf{x}^*)$ at moment of pick")
ax.set_title("Greedy picks always the most uncertain point; "
             "Boltzmann lets it slip; random ignores it")
ax.legend(fontsize=8, loc="lower left")
ps.apply_axis_style(ax)
plt.tight_layout()
ps.save_plot(fig, "04_U_at_pick", folder=str(OUT_DIR))
plt.close(fig)


# ── Save selection CSVs ------------------------------------------------------
for f in T_FRACS:
    name = ("greedy_T0"   if f == 0.0
            else "random_Tinf" if not np.isfinite(f)
            else f"boltz_T{f:g}sigma")
    s   = results[f]["s"]
    df  = pool.iloc[s].copy().reset_index(drop=True)
    df["pool_idx"]  = s
    df["U_at_pick"] = results[f]["u_pick"]
    df.to_csv(OUT_DIR / f"selected_{name}.csv", index=False)

print(f"\nAll outputs in: {OUT_DIR}")
print(f"Total runtime : {(time.perf_counter()-t0_all)/60:.1f} min")
