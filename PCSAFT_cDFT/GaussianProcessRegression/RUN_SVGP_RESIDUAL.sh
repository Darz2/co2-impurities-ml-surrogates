#!/bin/bash
#SBATCH --job-name=ML-SVGP_residual
#SBATCH --partition=parallel-short
#SBATCH --time=12:00:00
#SBATCH --exclude=c171
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

set -euo pipefail

start_time=$(date +"%T")
echo "Job started at: $start_time"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

echo "Python:  $(which python)"
echo "LaTeX :  $(which latex || echo 'latex not found')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPyTorch: $(python -c 'import gpytorch; print(gpytorch.__version__)')"

OUTPUT_DIR="SLURM_SVGP_residual"
mkdir -p "${OUTPUT_DIR}"

papermill SVGP_GPR_RESIDUAL.ipynb "${OUTPUT_DIR}/SVGP_residual_output.ipynb" \
    -p OUTPUT_FOLDER "${OUTPUT_DIR}" \
    -p SEED          4555525         \
    -p N_INDUCING    500             \
    -p N_EPOCHS      100             \
    -p BATCH_SIZE    1024            \
    -p LR            0.01            \
    -p RUN_CV        True            \
    -p CV_EPOCHS     30

echo "Job finished at: $(date +"%T")"
