#!/bin/bash
#SBATCH --job-name=SVGP_residual
#SBATCH --partition=highmem
#SBATCH --time=24:00:00
#SBATCH --nodelist=c109
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=4G

set -euo pipefail

# Use all allocated CPUs for BLAS/LAPACK (Cholesky in GPyTorch/SVGP)
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

echo "Python:  $(which python)"
echo "LaTeX :  $(which latex || echo 'latex not found')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "GPyTorch: $(python -c 'import gpytorch; print(gpytorch.__version__)')"

OUTPUT_DIR="SLURM_SVGP_residual"
mkdir -p "${OUTPUT_DIR}"

papermill SVGP_GPR_RESIDUAL.ipynb "${OUTPUT_DIR}/SVGP_residual_output.ipynb" \
    -p OUTPUT_FOLDER   "${OUTPUT_DIR}" \
    -p SEED            4555525         \
    -p N_INDUCING      1500            \
    -p N_INDUCING_TAIL 0.30            \
    -p N_EXTREME_TAIL  0.10            \
    -p TAIL_THRESHOLD  0.3             \
    -p TAIL_OVERSAMPLE 5               \
    -p N_EPOCHS        500             \
    -p BATCH_SIZE      1024            \
    -p LR              0.01            \
    -p LR_MIN          0.0001          \
    -p RUN_CV          True            \
    -p CV_EPOCHS       100

echo "Job finished at: $(date +"%T")"
