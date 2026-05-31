# cdft_vs_ml_bundle — portable timing/accuracy comparison

Self-contained. Copy this whole folder to an HPC with the `py_a6`/TabPFN env
(needs `torch`, `tabpfn`, `tabpfn_extensions`, `joblib`, `scikit-learn`, `pandas`).
No `feos` and no `SENSITIVITY_ANALYSIS` required.

    scp -r cdft_vs_ml_bundle  user@hpc:/path/    # then run the notebook on a GPU node

Run `cdft_vs_ml_timing.ipynb` (py_a6 kernel) top-to-bottom. Inference-only -> minutes.

inputs/
  model.joblib       with-HPO TabPFN gamma model
  feed_{0..4}.csv    cDFT (T,P,z) points + true gamma per feed
  cdft_times.json    features, target, per-feed cDFT map durations [s]
outputs (written here): cdft_vs_ml_timing.csv, cdft_vs_ml_table.tex
