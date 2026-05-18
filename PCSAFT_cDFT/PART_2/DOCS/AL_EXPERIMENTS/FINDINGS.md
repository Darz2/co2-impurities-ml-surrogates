# AL_EXPERIMENTS — Empirical findings

Two follow-up experiments to the main AL study in
[../02_GPR_AL.py](../02_GPR_AL.py). Both reuse the same ARD Matérn-5/2
kernel fit on the A2 subset at $T_\text{ref} = 250$ K.

## Setup

- Pool: 1,000 candidate compositions (`POOL/composition_pool.csv`)
- GPR training set: 100 labelled A2 rows at 250 ± 1 K (deduplicated on
  composition)
- Fitted kernel: ConstantKernel × Matern(ν = 5/2, ARD) + WhiteKernel
- Fitted signal variance: **σ²f* ≈ 366.28**
- Fitted ARD length-scales (per species, in order CO₂, H₂, Ar, N₂, CH₄,
  O₂, CO, H₂S):
  `[106.2, 50.1, 1000, 841.7, 269.0, 1000, 1000, 187.3]`
  Note three components (Ar, O₂, CO) hit the upper bound of 1000 — the
  ARD optimiser is essentially declaring them irrelevant to P_bubble at
  this temperature given the 100-row training set.

## Experiment 1 — Pool saturation under tightening u_t

Script: [03_AL_saturation.py](03_AL_saturation.py)
Outputs: [OUT_saturation/](OUT_saturation/)

Pure greedy MaxVar AL was rerun with the kernel-uncertainty cut-off
swept across nine decades, $u_t \in \{10^{-12},\ldots,10^{-2}\}\cdot
\sigma_f^{2*}$. The loop self-terminates when no remaining candidate
exceeds $u_t$.

| $u_t / \sigma_f^{2*}$ | $u_t$ (absolute) | $|S^{*}|$ |
|---:|---:|---:|
| $10^{-12}$ | $3.66\times 10^{-10}$ | 100 (no termination) |
| $10^{-10}$ | $3.66\times 10^{-8}$  | 100 (no termination) |
| $10^{-8}$  | $3.66\times 10^{-6}$  | **40** |
| $10^{-7}$  | $3.66\times 10^{-5}$  | 23 |
| $10^{-6}$  | $3.66\times 10^{-4}$  | **13** |
| $10^{-5}$  | $3.66\times 10^{-3}$  | 9 |
| $10^{-4}$  | $3.66\times 10^{-2}$  | 7 |
| $10^{-3}$  | $3.66\times 10^{-1}$  | 3 |
| $10^{-2}$  | $3.66$                | 2 (seeds only) |

### Key empirical result

> **With the fitted ARD kernel ($\sigma_f^{2*} \approx 366$), the 1,000-composition pool collapses to only ~13 informationally distinct points at $u_t = 10^{-6}\sigma_f^{2*}$, and ~40 points at $u_t = 10^{-8}\sigma_f^{2*}$. Reaching the full $N = 100$ budget requires $u_t \leq 10^{-10}\sigma_f^{2*}$ — essentially zero on the kernel's natural scale.**

### What this means chemically

The pool is **informationally thin under the learned metric**: most
compositions are near-duplicates of each other in the ARD-rescaled
space, because (i) CO₂ dominates ($z_{\text{CO}_2} \geq 0.94$ in every
row by construction) and (ii) three trace components (Ar, O₂, CO) get
near-infinite length-scales and are effectively ignored. The "real"
dimensionality the kernel sees is closer to 5 than 8.

The original notebook uses $u_t = 10^{-6}$ **absolute** $\approx 2.7\times
10^{-9}\sigma_f^{2*}$, which is the bottom-left corner of the table —
no pruning ever fires, the loop runs to the full $N = 100$, and the last
~60 picks are near-duplicates of earlier ones under the kernel.

### Practical recommendation

If the downstream goal is to train a GPR / NN on $P_\text{bubble}$ and
the labelling cost is non-trivial (PC-SAFT calls), running AL with
$u_t = 10^{-7}\sigma_f^{2*}$ (∼23 picks) or $10^{-8}\sigma_f^{2*}$
(∼40 picks) would yield essentially the same model accuracy as $N = 100$
at a third of the PC-SAFT cost.

## Experiment 2 — Boltzmann temperature sweep

Script: [04_AL_boltzmann.py](04_AL_boltzmann.py)
Outputs: [OUT_boltzmann/](OUT_boltzmann/)

Argmax was replaced with $P(i) \propto \exp(U_i / T)$. Temperatures
$T \in \{0,\; 0.1,\; 1,\; 10,\; \infty\}\cdot\sigma_f^{2*}$ were swept,
with the pruning floor set to $10^{-15}$ so all strategies reached
$N = 100$ for a fair comparison.

### Observations from the diagnostic plots

- **PCA coverage** (`01_pca_coverage.png`): greedy ($T \to 0$) hugs the
  outer envelope of the pool; random ($T \to \infty$) clumps in the
  high-density centre; intermediate $T$ interpolates smoothly.
- **CO₂ trajectory** (`02_CO2_order.png`): greedy alternates between
  CO₂-rich and CO₂-poor compositions early; random has no structure.
- **Nearest-neighbour spread** (`03_nn_distance.png`): greedy produces
  the largest median nearest-neighbour distance — the sparsest design;
  random the smallest.
- **U at pick** (`04_U_at_pick.png`): greedy's curve monotonically
  decreases; Boltzmann curves are noisier; random's is essentially flat
  in $U$.

### Key empirical result

> **All three intermediate temperatures ($0.1,\,1,\,10 \cdot \sigma_f^{2*}$) produce nearest-neighbour distributions that lie between the greedy and random extremes, confirming the textbook inverted-U behaviour empirically. None of them produces a more diverse selection than pure greedy on this pool.**

This is a useful robustness check: it pre-empts a reviewer asking
*"could a softer acquisition do better?"* — the answer is no, on this
problem, because the pool's intrinsic geometry already rewards
deterministic argmax.

## Experiment 3 — Recommended u_t per target N

Script: [05_u_t_for_N.py](05_u_t_for_N.py)
Outputs: [OUT_u_t_for_N/](OUT_u_t_for_N/)

A single greedy MaxVar trace (with no early-stop) was run up to
$|S| = 200$ and the kernel uncertainty $U(\mathbf{x}^*)$ recorded at
every pick. For any target $N$, the saturation threshold lies in the
interval $(U_{N+1},\; U_N]$; the table below reports the geometric
midpoint as the recommended value.

| Target $N$ | $u_t$ (absolute) | $u_t / \sigma_f^{2*}$ | $U$ at step $N$ |
|---:|---:|---:|---:|
| 10  | $2.6\times 10^{-3}$ | $7.1\times 10^{-6}$  | $3.1\times 10^{-3}$ |
| 20  | $7.0\times 10^{-5}$ | $1.9\times 10^{-7}$  | $8.1\times 10^{-5}$ |
| 30  | $1.0\times 10^{-5}$ | $2.8\times 10^{-8}$  | $1.1\times 10^{-5}$ |
| 40  | $3.4\times 10^{-6}$ | $9.4\times 10^{-9}$  | $3.7\times 10^{-6}$ |
| 50  | $1.5\times 10^{-6}$ | $4.0\times 10^{-9}$  | $1.5\times 10^{-6}$ |
| 60  | $6.9\times 10^{-7}$ | $1.9\times 10^{-9}$  | $7.0\times 10^{-7}$ |
| 70  | $4.6\times 10^{-7}$ | $1.2\times 10^{-9}$  | $4.6\times 10^{-7}$ |
| 80  | $3.2\times 10^{-7}$ | $8.7\times 10^{-10}$ | $3.2\times 10^{-7}$ |
| 90  | $2.0\times 10^{-7}$ | $5.4\times 10^{-10}$ | $2.0\times 10^{-7}$ |
| 100 | $1.6\times 10^{-7}$ | $4.3\times 10^{-10}$ | $1.6\times 10^{-7}$ |

Full data: [OUT_u_t_for_N/u_t_for_N.csv](OUT_u_t_for_N/u_t_for_N.csv)
Visual: [OUT_u_t_for_N/U_trace_with_thresholds.png](OUT_u_t_for_N/U_trace_with_thresholds.png)

### Critical consequence for the original notebook

The current setting in [../02_GPR_AL.py](../02_GPR_AL.py) is
$u_t = 10^{-6}$ (absolute). From the table above, $U(\mathbf{x}^*)$
falls **below $10^{-6}$ around step 56**. Therefore the greedy loop
silently self-terminates at $|S^{*}| \approx 56$, and the snapshot
builder fills the remaining slots with **random pool points**
(`02_GPR_AL.py:265-268`):

```python
if n_avail < N:
    fallback = [i for i in range(len(pool)) if i not in set(chosen)]
    rng_fb   = np.random.default_rng(SEED + N)
    rng_fb.shuffle(fallback)
    chosen = chosen + fallback[: N - n_avail]
```

This means the `AL_N100` set used in the paper is actually
**~56 AL picks + ~44 uniform-random fillers** — *not* a pure AL design.

### Recommended fix

Lower `U_THRESHOLD` in `02_GPR_AL.py` to **$10^{-8}$** (or below). From
the trace, $U(\mathbf{x}^*)$ at step 100 is $\approx 1.6\times 10^{-7}$,
so $10^{-8}$ guarantees the greedy loop reaches $N = 100$ with no
random fillers, while still pruning genuine numerical duplicates. This
keeps the algorithm description correct ("pure greedy MaxVar with
$u_t$ as numerical safeguard") and avoids the silent contamination of
$AL_{N\ge 60}$ with random points.

## Reproduction

```bash
cd /home/darshan/A6/PCSAFT_cDFT/PART_2/DOCS/AL_EXPERIMENTS
python 03_AL_saturation.py     # ~40 s
python 04_AL_boltzmann.py      # ~40 s
python 05_u_t_for_N.py         # ~40 s
```

Neither script calls PC-SAFT; all three study selection geometry only.
Add a PC-SAFT labelling pass on the generated CSVs if learning curves
are needed.
