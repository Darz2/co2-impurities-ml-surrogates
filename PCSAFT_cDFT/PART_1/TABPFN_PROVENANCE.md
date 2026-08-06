# TabPFN run provenance and Table S5 correction

**Date:** 2026-08-06
**Scope:** reconciling the TabPFN models reported in the manuscript, shipped in the
Supporting Information, and described in SI Table S5.

---

## 1. Summary

Three separate TabPFN hyperparameter-optimization (HPO) runs existed on disk. Before
this correction, three different artefacts of the submission each referred to a
*different* one of them:

| Artefact | Was pointing at | Should point at |
|---|---|---|
| Paper Table 3, `TabPFN`<sup>b</sup> (with HPO) + abstract + §3.6 | run **A** | run A ✅ (already correct) |
| Shipped supplementary model files | run **B** | run A ❌ |
| SI Table S5 hyperparameters | run **C** | run A ❌ |

Run **A** was made canonical, because the paper's headline numbers already quoted it
and it is the best-performing of the three. The shipped `.joblib`/`.csv` files were
replaced with run A copies, and Table S5 was rewritten from run A.

The paper's `TabPFN`<sup>a</sup> (no HPO) rows, Figure 4, and the parity plots come
from the separate `without_HPO` run and were always self-consistent — they were not
affected.

---

## 2. Run locations

All under `/home/darshan/A6/PCSAFT_cDFT/PART_1/`:

| | Path | Version | Architectures (bubble, dew, γ, thickness) | *P*<sub>bubble</sub> CV RMSE / [bar] | Role |
|---|---|---|---|---|---|
| **A** | `TabPFN-snellius-home/with_HPO/` | **v2.5** | single, dt_pfn, dt_pfn, single | **1.313 ± 0.302** | **canonical** — paper Table 3<sup>b</sup>, shipped models, Table S5 |
| **B** | `TabPFN-snellius-home/TabPFN_V3/with_HPO/` | **v2.5** | single ×4 | 3.093 ± 0.478 | previously shipped, now replaced |
| **C** | `TabPFN/with_HPO/OUTPUTS/` | **v2.5** | dt_pfn ×4 | 3.030 ± 0.585 | source of the old Table S5 |
| — | `TabPFN-snellius-home/without_HPO/` | `auto` | single ×4 | 2.732 ± 0.587 | paper Table 3<sup>a</sup>, Figure 4, parity plots |

Shipped location:
`/home/darshan/A6/OverleafDir/submission/SUPPLEMENTARY_MATERIALS/MACHINE_LEARNING/TabPFN/`
with subdirectories `P_bubble/`, `P_dew/`, `gamma/`, `interfacial_thickness/`.

### Subdirectory naming (differs between runs)

| Run | Naming convention |
|---|---|
| A, B | `SLURMBubble`, `SLURMDew`, `SLURMGamma`, `SLURMTHICKNESS` |
| C | `SLURM_Pbubble`, `SLURM_Pdew`, `SLURM_gamma`, `SLURM_thickness` |
| without_HPO | `TabPFN_<target>_OUTPUTS` |

Run B additionally contains `*_Verify` twins (e.g. `SLURMBubble_Verify`) with different
checksums and metrics (*P*<sub>bubble</sub> CV RMSE 3.200 ± 0.334). The plain
`SLURMBubble` variant is the one whose files had been shipped.

### Ruled out

Swept and excluded as sources of any submitted artefact:

- `PCSAFT_cDFT/ML/TabPFN/` — only 3 targets, CV RMSEs 2.611/0.822/0.155 and
  0.322/0.816/0.160, no checksum matches.
- `PCSAFT_cDFT/ML/TabPFN/OLD_TabPFN_RUNS/` — no CV block at all.

Exactly three HPO runs are in play.

---

## 3. TabPFN version

The manuscript originally claimed **v2.6**. It is **v2.5**.

Evidence: the resolved checkpoint filename was grepped out of every artefact in every
run. All three HPO runs load exclusively `tabpfn-v2.5-regressor-v2.5_*.ckpt` from the
Snellius cache (`/home/draju/.cache/tabpfn/`). Zero v2.6 strings anywhere on disk.

The decisive argument is structural rather than circumstantial: `model_path` was itself
an HPO search dimension over **seven** checkpoint variants — default, low-skew,
quantiles, real, real-variant, small-samples, variant. That is exactly the v2.5
regressor family. TabPFN v2.6 ships only `default`, so a search over seven variants is
only possible within v2.5. Any HPO'd model is therefore necessarily v2.5.

**Where the "v2.6" claim came from:** the local cache `~/.cache/tabpfn/` does contain
`tabpfn-v2.6-regressor-v2.6_default.ckpt`, downloaded 2026-04-30 14:59 — about 19
minutes before the seven v2.5 variants at 15:18. That file exists on the laptop but no
run ever loaded it, and the runs executed on Snellius against a different cache.

**Version claim updated to v2.5 in:**

- `OverleafDir/A6-Draft-Overleaf/paper.tex:435` — "implemented using the TabPFN-v2.5 Python package"
- `OverleafDir/Supporting-Information-A6/si.tex` — Table S5 caption, `\acf{tabpfn} (v2.5)`

### Caveat on the `without_HPO` run

Those four models carry the literal string `model_path = 'auto'`, i.e. no explicit
checkpoint was recorded. Their version is *inferred* from the environment, not proven
from the artefact, unlike the three HPO runs where the path is written out in full.

### Open item

`tabpfn_extensions` is **0.3.0** in the local environment (`py_A6`), while the paper
cites **v0.4** on the same line as the TabPFN version. Worth confirming what was
installed on Snellius before submission.

---

## 4. Where the old Table S5 numbers came from

The old values were **not** invented or mistyped — they are a faithful transcription of
a real Optuna run, just the wrong one. They trace to run **C**, specifically the four
`TabPFN_<target>_best_config.json` files under
`PCSAFT_cDFT/PART_1/TabPFN/with_HPO/OUTPUTS/`.

Decoding those four JSONs against the old table matches on **11 of 11 rows across all
4 columns — 44 of 44 cells**:

| Row | run C JSON key → value | Old table said |
|---|---|---|
| Architecture | `model_type = 1` (all 4) | dt_pfn ×4 ✅ |
| Checkpoint | `model_path` = 4 / 1 / 5 / 5 | real, low-skew, small-samples, small-samples ✅ |
| Tree depth | `max_depth = 3` (all 4) | 5, 5, 5, 5 ✅ |
| Average logits | `average_before_softmax` = 1 / 0 / 1 / 1 | No, Yes, No, No ✅ |
| Softmax temperature | `softmax_temperature = 0` (all 4) | 0.75 ×4 ✅ |
| "RF preprocessing" | `FINGERPRINT_FEATURE` = 0 / 1 / 0 / 0 | Yes, No, Yes, Yes ✅ |
| Outlier removal | `OUTLIER_REMOVAL_STD = 1` (all 4) | 7.0 ×4 ✅ |
| Min unique for numerical | `MIN_UNIQUE_FOR_NUMERICAL_FEATURES` = 1 / 0 / 2 / 0 | 5, 1, 10, 1 ✅ |
| Target preprocessing | `REGRESSION_Y_PREPROCESS_TRANSFORMS` = 1 / 0 / 2 / 0 | id.+SP, id., SP, id. ✅ |
| Polynomial features | `POLYNOMIAL_FEATURES = 0` | No ×4 ✅ |
| Ensemble size | `n_estimators = 0` | 4 ✅ |

(Column order throughout: *P*<sub>bubble</sub>, *P*<sub>dew</sub>, γ, thickness.)

### The tell

Run C pinned four categoricals at one end of their range in *every* column: depth 5,
temperature 0.75, outlier 7.0, and dt_pfn throughout. That degree of uniformity is the
signature of a run whose Optuna sampler never really explored — plausibly reading a
stale or truncated study. It also scored worst of the three.

---

## 5. Optuna encoded-config decode map

`best_config.json` stores integer indices into the search space, not values. Verified
mapping:

| Key | Index → value |
|---|---|
| `model_type` | 0 = single, 1 = dt_pfn |
| `model_path` | 0 = default, 1 = low-skew, 2 = quantiles, 3 = real-variant, 4 = real, 5 = small-samples, 6 = variant |
| `max_depth` | 0 = 2, 1 = 3, 2 = 4, 3 = 5 |
| `softmax_temperature` | 0 = 0.75, 1 = 0.80, 2 = 0.90, 3 = 0.95, 4 = 1.00, 5 = 1.05 |
| `OUTLIER_REMOVAL_STD` | 0 = ∞, 1 = 7.0, 2 = 12.0 |
| `MIN_UNIQUE_FOR_NUMERICAL_FEATURES` | 0 = 1, 1 = 5, 2 = 10, 3 = 30 |
| `REGRESSION_Y_PREPROCESS_TRANSFORMS` | 0 = id., 1 = id.+SP, 2 = SP |
| `average_before_softmax` | 0 = Yes (True), 1 = No (False) |
| `FINGERPRINT_FEATURE` | 0 = Yes (True), 1 = No (False) |
| `n_estimators` | 0 = 4 |
| `POLYNOMIAL_FEATURES` | 0 = No |
| `PREPROCESS_TRANSFORMS` | index into 180 transform combinations |

Note the **inverted** sense of `average_before_softmax` and `FINGERPRINT_FEATURE`:
index 0 means `True`.

**Schema difference:** run A and run C store the indices under the key `best_config`;
run B stores them under `encoded_best_params` and additionally writes a decoded
`model_params` block with the resolved checkpoint path and inference config.

---

## 6. Architecture across runs

`dt_pfn` for all four targets occurs **only** in run C — the anomaly, not the norm.

| | *P*<sub>bubble</sub> | *P*<sub>dew</sub> | γ | Thickness |
|---|---|---|---|---|
| **A** (canonical) | single | **dt_pfn** (depth 4) | **dt_pfn** (depth 2) | single |
| B | single | single | single | single |
| C | **dt_pfn** (5) | **dt_pfn** (5) | **dt_pfn** (5) | **dt_pfn** (5) |

Verified two independent ways — decoding `best_config.json`, and loading the shipped
`.joblib` files and reading their Python class:

```
P_bubble               TabPFNRegressor
P_dew                  DecisionTreeTabPFNRegressor
gamma                  DecisionTreeTabPFNRegressor
interfacial_thickness  TabPFNRegressor
```

### Why this matters for the prediction-intervals section

The new SI subsection argues that *P*<sub>bubble</sub> gets native TabPFN prediction
intervals while γ requires the WSD + SVGP route, because the model selected for γ is
the decision-tree architecture whose aggregated output has no well-defined quantiles.

That reasoning holds under run A — γ is dt_pfn, *P*<sub>bubble</sub> is single, exactly
the asymmetry the argument needs. Under the old run C it would have been **false**:
every model there was dt_pfn, so the same argument would have ruled out TabPFN
intervals for *P*<sub>bubble</sub> too. The correction made that section
self-consistent.

### Why dt_pfn cannot emit prediction intervals

1. The per-node call is `self.tabpfn.predict(X_subset)` with no `output_type` or
   `quantiles` argument — each node's predictive density is discarded at the point of
   use.
2. The regression merge is a running arithmetic mean of point predictions
   (`y_prob_averaging[...] += leaf_prediction[...]` then `/= 2.0`). A mean of means
   carries no distributional information.
3. With `adaptive_tree=True` the final estimator is a data-dependent composition of
   averaging / replacement / previous-value decisions.

`predict()` is typed `-> np.ndarray` and the regressor exposes no `predict_proba`.
Patchable in principle (~100–200 lines: request quantiles per node, replay the node
decisions on the densities, re-bin onto a common grid, invert the mixture CDF), but
that would be an unvalidated construction — hence the SVGP route for γ.

---

## 7. Changes made

### 7.1 Shipped model files (8 files replaced)

```bash
SRC=/home/darshan/A6/PCSAFT_cDFT/PART_1/TabPFN-snellius-home/with_HPO
DST=/home/darshan/A6/OverleafDir/submission/SUPPLEMENTARY_MATERIALS/MACHINE_LEARNING/TabPFN
for pair in "SLURMBubble:P_bubble" "SLURMDew:P_dew" "SLURMGamma:gamma" "SLURMTHICKNESS:interfacial_thickness"; do
  d="${pair%%:*}"; t="${pair##*:}"
  cp -f "$SRC/$d/TabPFN_${t}_model.joblib"    "$DST/$t/TabPFN_${t}_model.joblib"
  cp -f "$SRC/$d/TabPFN_${t}_predictions.csv" "$DST/$t/TabPFN_${t}_predictions.csv"
done
```

All 8 md5-verified identical to source. Backup of the replaced run B files:
`scratchpad/TabPFN_submitted_backup_V3/`.

### 7.2 SI Table S5 (`si.tex`, label `tab:tabpfn_hpo`)

Body rewritten from run A. Roughly 8 of 12 rows differed per column. Additionally:

- Row **"RF preprocessing" renamed to "Fingerprint feature"** — it was always the
  `FINGERPRINT_FEATURE` inference-config flag. It is the only search-space key left
  unassigned once the other 11 rows are mapped.
- Its **Default corrected No → Yes**; the package default is `True`
  (`tabpfn/inference_config.py:101`).
- Caption abbreviation list extended with Quant-norm, Squash, and None.

### 7.3 Version claim

`paper.tex:435` and the Table S5 caption in `si.tex`, both v2.6 → v2.5. The
TabPFN-extensions version on the same line as the former was deliberately left
untouched (different package — see the open item in §3).

---

## 8. Verification

- **Table S5 vs shipped models:** hyperparameters read back out of the now-shipped
  `.joblib` files and compared row by row against the typeset table — match.
- **Compilation:** both `latexmk` runs exit 0. Final-pass logs clean — no errors, no
  undefined references or citations, no multiply-defined labels. `paper.pdf` 81 pages,
  `si.pdf` 69 pages.
- **Overfull boxes:** SI has zero. The paper has three, all confirmed pre-existing via
  `git diff -U0` (1.57 pt at line 540; **38.68 pt at line 586**; 2.94 pt at lines
  666–677). Line 586 is the long subsubsection heading "Physically interpretable SR
  correction for the WSD semi-empirical correlation" — too long to break, left as is by
  decision.
- **Visual check:** SI page 36 (Table S5) and page 40 (Figure S27, all three panels plus
  caption) rendered and inspected.
- **SI figures:** S1–S34 present and contiguous.

---

## 9. Useful commands

Loading GPU-pickled joblib models on a CPU-only machine:

```python
import functools, torch, joblib
torch.load = functools.partial(torch.load, map_location="cpu")
model = joblib.load(path)
```

Do **not** `repr()` all model attributes — `RegressorEnsembleConfig` raises
`AttributeError: 'RegressorEnsembleConfig' object has no attribute 'subsample_ix'`.
Extract an explicit key list plus `inference_config` instead.

Environment with all required packages (docx, PIL, gpytorch, torch, tabpfn,
tabpfn_extensions): `/home/darshan/A6/py_A6/bin/python`. The system `python` lacks
python-docx.

Grepping the checkpoint version out of a run directory:

```bash
grep -rhoa "tabpfn-v2\.[0-9]*-regressor[-a-z0-9_.]*\.ckpt" <run_dir> | sort | uniq -c
```

---

## 10. State

Nothing has been committed in `A6`, `A6-Draft-Overleaf`, or `Supporting-Information-A6`
as part of this work; `git diff` shows everything. (The SI repo has since picked up an
automatic `UPDATE - 2026-08-06 14:52` commit, `c4baaf5`, which contains the Table S5
rewrite alongside the prediction-intervals subsection.)
