"""
05_u_t_for_N.py
───────────────
Compute the u_t value that would cause the greedy MaxVar AL loop to
self-terminate at exactly |S*| = N, for every N in a chosen grid.

Method: run greedy AL once with no early-stop, record U at every pick,
then for target N the saturation threshold is the U value at step N+1
(any u_t in the interval (U[N+1], U[N]] yields |S*| = N).
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

import thermoift.PLOT_SETTINGS as ps

HERE      = Path(__file__).resolve().parent
PART2     = HERE.parent.parent
POOL_PATH = PART2 / "POOL" / "composition_pool.csv"
A2_PATH   = PART2.parent / "DATASET_A2" / "CombinedDataset_A2.csv"
OUT_DIR   = HERE / "OUT_u_t_for_N"
OUT_DIR.mkdir(exist_ok=True)

T_REF, SEED, GPR_MAX, N_RESTART, N_MAX = 250.0, 770077, 5000, 3, 200
CSV_TO_FULL = {"CO2":"carbon dioxide","H2":"hydrogen","Ar":"argon","N2":"nitrogen",
               "CH4":"methane","O2":"oxygen","CO":"carbon monoxide","H2S":"hydrogen sulfide"}
Z_FEATURES = [f"z_{v}" for v in CSV_TO_FULL.values()]
COMP_MAP   = {k: f"z_{v}" for k, v in CSV_TO_FULL.items()}
TARGET     = "P_bubble"

# ── Data + GPR fit (same as 03/04) ───────────────────────────────────────────
pool = pd.read_csv(POOL_PATH)
a2   = pd.read_csv(A2_PATH); a2.columns = [c.strip() for c in a2.columns]
extra = [c for c in a2.columns if c.startswith("z_") and c not in Z_FEATURES]
if extra:
    a2 = a2[a2[extra].sum(axis=1) < 1e-6].copy()
a2 = a2[np.abs(a2["temperature"] - T_REF) <= 1.0]
df = (a2[Z_FEATURES + [TARGET]]
        .drop_duplicates(subset=Z_FEATURES)
        .dropna(subset=[TARGET]).reset_index(drop=True))
if len(df) > GPR_MAX:
    df = df.sample(GPR_MAX, random_state=SEED).reset_index(drop=True)

n_feat = len(Z_FEATURES)
kernel = (ConstantKernel(1.0, (1e-3, 1e3))
          * Matern(nu=2.5, length_scale=np.ones(n_feat),
                   length_scale_bounds=(1e-3, 1e3))
          + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 100.0)))
gpr = Pipeline([("scaler", StandardScaler()),
                ("gpr", GaussianProcessRegressor(
                    kernel=kernel, alpha=0.0, normalize_y=True,
                    n_restarts_optimizer=N_RESTART, random_state=SEED))])
gpr.fit(df[Z_FEATURES], df[TARGET])
scaler        = gpr.named_steps["scaler"]
signal_kernel = gpr.named_steps["gpr"].kernel_.k1
sigma_f_sq    = float(signal_kernel.k1.constant_value)
X_pool        = scaler.transform(pool.rename(columns=COMP_MAP)[Z_FEATURES].values)
print(f"σ_f²* = {sigma_f_sq:.4f}")


# ── Greedy AL with no termination, record U at every pick ────────────────────
def greedy_trace(X_all, kern, n_max, seed):
    rng    = np.random.default_rng(seed)
    n_pool = len(X_all)
    s_list = list(rng.choice(n_pool, size=2, replace=False))
    p_set  = set(range(n_pool)) - set(s_list)
    u_pick = [np.nan, np.nan]
    while p_set and len(s_list) < n_max:
        p_list = list(p_set)
        K_TT   = kern.diag(X_all[p_list])
        K_TS   = kern(X_all[p_list], X_all[s_list])
        K_SS   = kern(X_all[s_list], X_all[s_list]) + 1e-8 * np.eye(len(s_list))
        L      = np.linalg.cholesky(K_SS)
        V      = np.linalg.solve(L, K_TS.T)
        U      = np.maximum(K_TT - np.einsum("ij,ij->j", V, V), 0.0)
        best   = int(np.argmax(U))
        s_list.append(p_list[best])
        u_pick.append(float(U[best]))
        p_set.discard(p_list[best])
    return s_list, u_pick

n_pool = len(X_pool)
print(f"Tracing greedy AL up to |S| = {min(N_MAX, n_pool)} …")
t0 = time.perf_counter()
s_list, u_pick = greedy_trace(X_pool, signal_kernel, min(N_MAX, n_pool), SEED)
print(f"  done in {time.perf_counter()-t0:.1f} s, picks = {len(s_list)}")


# ── Build N → u_t table ──────────────────────────────────────────────────────
# For target N, the threshold must satisfy U[N] > u_t >= U[N+1].
# We report a "safe" u_t = mean(U[N], U[N+1]) on a log scale.
u_arr = np.asarray(u_pick)            # u_arr[k] = U at step (k+1)
N_grid = list(range(10, 110, 10)) + [120, 150, 200]
N_grid = [N for N in N_grid if N + 1 < len(u_arr)]

rows = []
for N in N_grid:
    u_lo   = u_arr[N]          # U at step (N+1) ; loop terminates if u_t >= this
    u_hi   = u_arr[N - 1]      # U at step N     ; loop must allow this pick
    u_safe = np.sqrt(u_lo * u_hi) if (u_lo > 0 and u_hi > 0) else u_lo
    rows.append({
        "target_N":           N,
        "u_t_absolute":       u_safe,
        "u_t_over_sigma_f2":  u_safe / sigma_f_sq,
        "U_at_step_N":        u_hi,
        "U_at_step_N_plus_1": u_lo,
    })
tbl = pd.DataFrame(rows)
tbl.to_csv(OUT_DIR / "u_t_for_N.csv", index=False)

print("\nRecommended u_t for each target N:")
print(tbl.to_string(index=False,
                    formatters={
                        "u_t_absolute":       "{:.3e}".format,
                        "u_t_over_sigma_f2":  "{:.3e}".format,
                        "U_at_step_N":        "{:.3e}".format,
                        "U_at_step_N_plus_1": "{:.3e}".format}))


# ── Plot: U(step) with horizontal lines at each chosen u_t -------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(1, len(u_arr) + 1), u_arr, "-o", ms=3, color="0.25",
        lw=1.2, label="U at each greedy pick")
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(tbl)))
for col, (_, r) in zip(cmap, tbl.iterrows()):
    ax.axhline(r["u_t_absolute"], color=col, ls=":", lw=1.0)
    ax.text(len(u_arr) + 1, r["u_t_absolute"], f"N={int(r['target_N'])}",
            color=col, fontsize=8, va="center")
ax.set_yscale("log")
ax.set_xlabel("greedy selection step")
ax.set_ylabel(r"$U(\mathbf{x}^*)$ at moment of pick")
ax.set_title(r"Greedy MaxVar U trajectory and the $u_t$ that stops it at each $N$")
ps.apply_axis_style(ax)
plt.tight_layout()
ps.save_plot(fig, "U_trace_with_thresholds", folder=str(OUT_DIR))
plt.close(fig)

print(f"\nOutputs in: {OUT_DIR}")
