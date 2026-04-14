#!/bin/bash
#SBATCH --job-name=ML-GPR_residual-4000
#SBATCH --partition=parallel-short
#SBATCH --time=06:00:00
#SBATCH --exclude=c171
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "=================================================="
echo "Samples    : 4000"
echo "Seed       : 444444444"
echo "Output dir : SLURM_GPR_residual_4000"
echo "CPUs       : ${SLURM_CPUS_PER_TASK}"
echo "Started at : $(date +"%T")"
echo "=================================================="

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

echo "Python : $(which python)"
echo "LaTeX  : $(which latex || echo 'latex not found')"

mkdir -p "SLURM_GPR_residual_4000"

papermill GPR.ipynb "SLURM_GPR_residual_4000/GPR_residual_output.ipynb" \
    -p OUTPUT_FOLDER        "SLURM_GPR_residual_4000" \
    -p SEED                 444444444                  \
    -p EXPERIMENT_MAX_SAMPLES 4000                     \
    -p RESTART_OPTIMIZER    10                         \
    -p STRAT_BINS_SAMPLING  10                         \
    -p STRAT_BINS_SPLIT     2                          \
    -p RUN_CV               False                      \
    -p CV_FOLDS             5

echo "Finished at: $(date +"%T")"
