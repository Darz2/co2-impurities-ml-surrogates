"""
03_AL_saturation.py
───────────────────
Experiment: tighten the kernel-uncertainty cut-off  u_t  and observe
that the greedy AL loop self-terminates when the candidate pool is
informationally saturated under the learned ARD metric.

What this script DOES NOT do:
  - it does not change the *order* of selection (still pure argmax)
  - it does not invoke PC-SAFT (selection geometry only)
  - it does not refit the GPR

What it DOES:
  - sweeps u_t across 6 orders of magnitude
  - records (i) the saturation point |S*| at which the loop stops,
              (ii) the U-trajectory of the selected points,
              (iii) the CO2 / mixture-size diversity of S*
  - produces 4 diagnostic plots

Run from this directory:
    python 03_AL_saturation.py
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

t0_all = time.perf_counter()

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent
PART2     = HERE.parent.parent
POOL_PATH = PART2 / "POOL" / "composition_pool.csv"
A2_PATH   = PART2.parent / "DATASET_A2" / "CombinedDataset_A2.csv"
OUT_DIR   = HERE / "OUT_saturation"
OUT_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
T_REF     = 250.0
N_MAX     = 100
SEED      = 770077
GPR_MAX   = 5000
N_RESTART = 3

# Threshold sweep — relative to σ_f²* (the kernel signal variance).
# A very wide log-grid is used because σ_f²* is large in this problem
# and U decays steeply along the selection trajectory.
U_T_FRACS = [1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

CSV_TO_FULL = {"CO2":"carbon dioxide","H2":"hydrogen","Ar":"argon","N2":"nitrogen",
               "CH4":"methane","O2":"oxygen","CO":"carbon monoxide","H2S":"hydrogen sulfide"}
POOL_Z_COLS = list(CSV_TO_FULL.keys())
Z_FEATURES  = [f"z_{v}" for v in CSV_TO_FULL.values()]
COMP_MAP    = {k: f"z_{v}" for k, v in CSV_TO_FULL.items()}
TARGET      = "P_bubble"


# ── Load pool & A2 training set ───────────────────────────────────────────────
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

print(f"  pool      : {len(pool):,} compositions")
print(f"  GPR train : {len(df_train):,} labelled rows at ~{T_REF} K")


# ── Fit GPR (same kernel as 02_GPR_AL.py) ─────────────────────────────────────
print("Fitting GPR …")
n_feat = len(Z_FEATURES)
kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * Matern(nu=2.5, length_scale=np.ones(n_feat), length_scale_bounds=(1e-3, 1e3))
    + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 100.0))
)
gpr = Pipeline([
    ("scaler", StandardScaler()),
    ("gpr",    GaussianProcessRegressor(
        kernel               = kernel,
        alpha                = 0.0,
        normalize_y          = True,
        n_restarts_optimizer = N_RESTART,
        random_state         = SEED,
    )),
])
t0 = time.perf_counter()
gpr.fit(df_train[Z_FEATURES], df_train[TARGET])
print(f"  fit time : {time.perf_counter()-t0:.1f} s")

scaler        = gpr.named_steps["scaler"]
fitted_k      = gpr.named_steps["gpr"].kernel_
signal_kernel = fitted_k.k1                          # ConstantKernel * Matern
sigma_f_sq    = float(signal_kernel.k1.constant_value)
ls_star       = np.asarray(signal_kernel.k2.length_scale)
print(f"  σ_f²*    = {sigma_f_sq:.4f}")
print(f"  ARD ℓ*   = {np.array2string(ls_star, precision=3)}")


# ── Pool features in the GPR's standardised space ────────────────────────────
pool_renamed = pool.rename(columns=COMP_MAP)
X_pool       = scaler.transform(pool_renamed[Z_FEATURES].values)


# ── AL with parametric u_t ----------------------------------------------------
def compute_U(X_T, X_S, kern):
    K_TT_diag = kern.diag(X_T)
    K_TS      = kern(X_T, X_S)
    K_SS      = kern(X_S, X_S) + 1e-8 * np.eye(len(X_S))
    L = np.linalg.cholesky(K_SS)
    V = np.linalg.solve(L, K_TS.T)
    return np.maximum(K_TT_diag - np.einsum("ij,ij->j", V, V), 0.0)


def run_al(X_all, kern, n_max, u_t, seed):
    """
    Pure greedy MaxVar AL with parametric u_t.
    Returns:
        s_list  : ordered selected pool indices
        u_picked: U value of each pick at the moment of selection
    """
    rng    = np.random.default_rng(seed)
    n_pool = len(X_all)
    s_list = list(rng.choice(n_pool, size=2, replace=False))
    p_set  = set(range(n_pool)) - set(s_list)
    u_picked = [np.nan, np.nan]            # seeds have no U

    while p_set and len(s_list) < n_max:
        p_list = list(p_set)
        U      = compute_U(X_all[p_list], X_all[s_list], kern)
        best   = int(np.argmax(U))

        if U[best] <= u_t:
            break                          # ← saturation
        s_list.append(p_list[best])
        u_picked.append(float(U[best]))
        p_set.discard(p_list[best])

        for i, idx in enumerate(p_list):
            if U[i] < u_t:
                p_set.discard(idx)

    return s_list, u_picked


# ── Run the sweep ─────────────────────────────────────────────────────────────
print("\nRunning u_t sweep …")
results = {}
for frac in U_T_FRACS:
    u_t = frac * sigma_f_sq
    t0  = time.perf_counter()
    s_list, u_pick = run_al(X_pool, signal_kernel, N_MAX, u_t, SEED)
    elapsed = time.perf_counter() - t0
    results[frac] = {"u_t": u_t, "s": s_list, "u_pick": u_pick, "time": elapsed}
    print(f"  frac={frac:7.0e}   u_t={u_t:.4e}   |S*|={len(s_list):3d}   "
          f"t={elapsed:5.2f}s")


# ── Plot 1: saturation curve |S*| vs u_t -------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
fracs   = list(results.keys())
sizes   = [len(results[f]["s"]) for f in fracs]
u_t_vals = [results[f]["u_t"]   for f in fracs]
ax.semilogx(u_t_vals, sizes, "-o", lw=2, ms=7, color="crimson")
ax.axhline(N_MAX, ls="--", color="0.4", lw=1, label=f"budget N={N_MAX}")
ax.set_xlabel(r"uncertainty cut-off $u_t$ (kernel units)")
ax.set_ylabel(r"saturation size $|S^{*}|$")
ax.set_title(r"Pool saturation under tightening $u_t$")
for f, x, y in zip(fracs, u_t_vals, sizes):
    ax.annotate(rf"${f:.0e}\,\sigma_f^{{2*}}$", (x, y),
                textcoords="offset points",
                xytext=(6, 6), fontsize=8, color="0.3")
ax.legend()
ps.apply_axis_style(ax)
plt.tight_layout()
ps.save_plot(fig, "01_saturation_curve", folder=str(OUT_DIR))
plt.close(fig)


# ── Plot 2: U trajectory of picks for each threshold -------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(fracs)))
for col, f in zip(cmap, fracs):
    u_pick = results[f]["u_pick"]
    ax.plot(range(1, len(u_pick) + 1), u_pick,
            color=col, lw=1.6,
            label=fr"$u_t = {f:.0e}\sigma_f^{{2*}}$  ($|S^*|={len(u_pick)}$)")
    ax.axhline(results[f]["u_t"], ls=":", color=col, lw=0.8)
ax.set_yscale("log")
ax.set_xlabel("selection step")
ax.set_ylabel(r"$U(\mathbf{x}^{*})$ at moment of pick")
ax.set_title("Kernel uncertainty of each pick along the AL trajectory")
ax.legend(fontsize=8, loc="upper right")
ps.apply_axis_style(ax)
plt.tight_layout()
ps.save_plot(fig, "02_U_trajectory", folder=str(OUT_DIR))
plt.close(fig)


# ── Plot 3: CO2 coverage of selected sets ------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2))
for col, f in zip(cmap, fracs):
    co2_vals = pool.iloc[results[f]["s"]]["CO2"].values
    y = np.full_like(co2_vals, f, dtype=float)
    ax.scatter(co2_vals, y, s=18, color=col, alpha=0.7, edgecolors="none")
ax.set_yscale("log")
ax.set_xlabel(r"CO$_2$ mole fraction in selected composition")
ax.set_ylabel(r"$u_t$ as fraction of $\sigma_f^{2*}$")
ax.set_yticks(fracs); ax.set_yticklabels([f"{f:.0e}" for f in fracs])
ax.set_title(r"Distribution of selected compositions in CO$_2$ space")
ps.apply_axis_style(ax)
plt.tight_layout()
ps.save_plot(fig, "03_CO2_coverage", folder=str(OUT_DIR))
plt.close(fig)


# ── Plot 4: mixture-size histogram of selected sets --------------------------
fig, ax = plt.subplots(figsize=(8, 4.2))
ms_levels = sorted(pool["mixture_size"].unique())
width     = 0.8 / len(fracs)
for j, (col, f) in enumerate(zip(cmap, fracs)):
    ms_sel = pool.iloc[results[f]["s"]]["mixture_size"].values
    counts = [np.sum(ms_sel == m) for m in ms_levels]
    xs     = np.arange(len(ms_levels)) + j * width
    ax.bar(xs, counts, width=width, color=col, edgecolor="white", linewidth=0.4,
           label=rf"${f:.0e}\,\sigma_f^{{2*}}$  ($|S|={len(results[f]['s'])}$)")
ax.set_xticks(np.arange(len(ms_levels)) + width * (len(fracs) - 1) / 2)
ax.set_xticklabels(ms_levels)
ax.set_xlabel("mixture size")
ax.set_ylabel("count in selected set")
ax.set_title(r"Mixture-size composition of the selected set vs $u_t$")
ax.legend(fontsize=7, ncol=2)
ps.apply_axis_style(ax)
plt.tight_layout()
ps.save_plot(fig, "04_mixture_size", folder=str(OUT_DIR))
plt.close(fig)


# ── Save per-threshold selection CSVs ----------------------------------------
for f in fracs:
    s = results[f]["s"]
    df = pool.iloc[s].copy().reset_index(drop=True)
    df["pool_idx"] = s
    df["U_at_pick"] = results[f]["u_pick"]
    df.to_csv(OUT_DIR / f"selected_u_t_{f:.0e}.csv", index=False)

print(f"\nAll outputs in: {OUT_DIR}")
print(f"Total runtime : {(time.perf_counter()-t0_all)/60:.1f} min")
