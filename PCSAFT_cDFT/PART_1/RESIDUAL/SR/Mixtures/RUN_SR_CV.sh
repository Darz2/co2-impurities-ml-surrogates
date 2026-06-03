#!/bin/bash
#SBATCH --job-name=SR_CV_mixtures
#SBATCH --partition=highmem
#SBATCH --nodelist=c109
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-SR_CV-%j.out

set -euo pipefail

echo "=================================================="
echo "Job started at : $(date +"%T")"
echo "Running on node: $(hostname)"
echo "CPUs allocated : ${SLURM_CPUS_PER_TASK}"
echo "=================================================="

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ── Environment ───────────────────────────────────────────────────────────────
export PATH="$HOME/.juliaup/bin:$PATH"
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

# Use every allocated core for the symbolic-regression population search,
# and keep BLAS single-threaded so it does not oversubscribe the cores.
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Number of SR iterations per fold (matches the production Mixture.jl runs).
# Override at submit time, e.g.:  sbatch --export=ALL,SR_CV_NITERATIONS=100 RUN_SR_CV.sh
export SR_CV_NITERATIONS=${SR_CV_NITERATIONS:-200}

echo "Julia          : $(which julia)"
echo "Julia version  : $(julia --version)"
echo "JULIA_NUM_THREADS = ${JULIA_NUM_THREADS}"
echo "SR_CV_NITERATIONS = ${SR_CV_NITERATIONS}"

# ── Run 5-fold CV for both feature sets (V1 and V2) ───────────────────────────
julia SR_cross_validation.jl

echo "=================================================="
echo "Job finished at: $(date +"%T")"
echo "Results in     : SR_CV_OUTPUTS/SR_CV_metrics.json"
echo "=================================================="
