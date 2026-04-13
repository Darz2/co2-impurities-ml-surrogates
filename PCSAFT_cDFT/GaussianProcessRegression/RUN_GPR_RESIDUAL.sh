#!/bin/bash
#SBATCH --job-name=ML-GPR_residual
#SBATCH --partition=parallel-short
#SBATCH --time=6:00:00
#SBATCH --exclude=c171
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G

set -euo pipefail

start_time=$(date +"%T")
echo "Job started at: $start_time"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

echo "Python: $(which python)"
echo "LaTeX : $(which latex || echo 'latex not found')"

OUTPUT_DIR="SLURM_GPR_residual"
mkdir -p "${OUTPUT_DIR}"

papermill GPR.ipynb "${OUTPUT_DIR}/GPR_residual_output.ipynb" \
    -p OUTPUT_FOLDER "${OUTPUT_DIR}" \
    -p SEED 4555525 \
    -p EXPERIMENT_MAX_SAMPLES 2000 \
    -p RESTART_OPTIMIZER 10 \
    -p RUN_CV False \
    -p CV_FOLDS 5

echo "Job finished at: $(date +"%T")"
