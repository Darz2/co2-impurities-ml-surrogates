#!/bin/bash
#SBATCH --job-name=clean_AL
#SBATCH --partition=highmem
#SBATCH --nodelist=c109
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=4:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=== ActiveLearning/clean.ipynb started at $(date '+%F %T') on $(hostname) ==="
echo "CPUs: ${SLURM_CPUS_PER_TASK}  (AL_ST + AL_MT processed together)"

papermill clean.ipynb clean_output.ipynb

echo "=== ActiveLearning/clean.ipynb finished at $(date '+%F %T') ==="
