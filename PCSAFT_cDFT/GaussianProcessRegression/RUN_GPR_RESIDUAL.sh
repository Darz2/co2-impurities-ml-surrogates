#!/bin/bash
#SBATCH --job-name=ML-GPR_residual-5000
#SBATCH --partition=parallel-short
#SBATCH --time=06:00:00
#SBATCH --exclude=c171
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# Use all allocated CPUs for BLAS/LAPACK (Cholesky in sklearn GPR)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

start_time=$(date +"%T")
echo "Job started at: $start_time"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

echo "Python: $(which python)"
echo "LaTeX : $(which latex || echo 'latex not found')"

OUTPUT_DIR="SLURM_GPR_residual_5000"
mkdir -p "${OUTPUT_DIR}"

papermill GPR.ipynb "${OUTPUT_DIR}/GPR_residual_output.ipynb" \
    -p OUTPUT_FOLDER "${OUTPUT_DIR}" \
    -p SEED 455676323 \
    -p EXPERIMENT_MAX_SAMPLES 5000 \
    -p RESTART_OPTIMIZER 10 \
    -p RUN_CV False \
    -p CV_FOLDS 5

echo "Job finished at: $(date +"%T")"
