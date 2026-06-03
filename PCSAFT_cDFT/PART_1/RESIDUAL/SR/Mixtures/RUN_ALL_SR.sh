#!/bin/bash
#SBATCH --job-name=SR_all_mixtures
#SBATCH --partition=highmem
#SBATCH --nodelist=c109
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-SR_all-%j.out

# ============================================================
#  Sequential pipeline for the SR mixtures study (V1 + V2):
#    1. Mixture.jl            -> SR_MIXTURES_OUTPUTS     (V1 production)
#    2. Mixture_V2.jl         -> SR_MIXTURES_OUTPUTS_V2  (V2 production)
#    3. SR_cross_validation.jl-> SR_CV_OUTPUTS           (5-fold CV, both)
#    4. make_sr_plots.py      -> parity + residual plots (both)
#
#  After this finishes, the report tables/equations are updated by hand
#  (Claude) and SR_Mixtures_Report.tex is recompiled.
#
#  Submit:  sbatch RUN_ALL_SR.sh
# ============================================================

set -euo pipefail

echo "=================================================="
echo "Job started at : $(date +'%F %T')"
echo "Node           : $(hostname)"
echo "CPUs allocated : ${SLURM_CPUS_PER_TASK}"
echo "=================================================="

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ── Environment ───────────────────────────────────────────────────────────────
export PATH="$HOME/.juliaup/bin:$PATH"
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"
source /home/darshan/A6/py_A6/bin/activate

# All cores for the SR population search; BLAS single-threaded to avoid oversubscription.
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# CV iterations per fold (matches production); override at submit time if desired.
export SR_CV_NITERATIONS=${SR_CV_NITERATIONS:-200}

echo "Julia  : $(julia --version)"
echo "Python : $(which python)"
echo "Threads: JULIA_NUM_THREADS=${JULIA_NUM_THREADS}  SR_CV_NITERATIONS=${SR_CV_NITERATIONS}"
echo

step () { echo; echo ">>> [$(date +'%T')] $*"; echo; }

# ── 1. V1 production run ───────────────────────────────────────────────────────
step "Step 1/4  V1 production  (Mixture.jl)"
julia Mixture.jl

# ── 2. V2 production run ───────────────────────────────────────────────────────
step "Step 2/4  V2 production  (Mixture_V2.jl)"
julia Mixture_V2.jl

# ── 3. 5-fold cross-validation (both versions) ────────────────────────────────
step "Step 3/4  5-fold cross-validation  (SR_cross_validation.jl)"
julia SR_cross_validation.jl

# ── 4. Parity + residual plots (both versions) ────────────────────────────────
step "Step 4/4  Post-processing plots  (make_sr_plots.py)"
python make_sr_plots.py

echo
echo "=================================================="
echo "All steps finished at: $(date +'%F %T')"
echo "Outputs:"
echo "  SR_MIXTURES_OUTPUTS/      (V1 equation, metrics, plots, CSVs)"
echo "  SR_MIXTURES_OUTPUTS_V2/   (V2 equation, metrics, plots, CSVs)"
echo "  SR_CV_OUTPUTS/            (SR_CV_metrics.json)"
echo "Next: update + recompile SR_Mixtures_Report.tex"
echo "=================================================="
