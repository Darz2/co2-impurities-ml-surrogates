#!/bin/bash
#SBATCH --job-name=ML-GPR_residual
#SBATCH --partition=highmem
#SBATCH --time=7-00:00:00
#SBATCH --nodelist=c109
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=51
#SBATCH --mem=64G
#SBATCH --array=0-4%5

set -euo pipefail

# ── Sample sizes (one per array task) ────────────────────────────────────────
SIZES=(1000 2000 3000 4000 5000)
N_SAMPLES=${SIZES[$SLURM_ARRAY_TASK_ID]}

# ── Per-run seeds (reproducible but independent across sizes) ─────────────────
SEEDS=(111111111 222222222 333333333 444444444 555555555)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUTPUT_DIR="SLURM_GPR_residual_${N_SAMPLES}"

# ── Threading ─────────────────────────────────────────────────────────────────
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "=================================================="
echo "Array task : ${SLURM_ARRAY_TASK_ID}"
echo "Samples    : ${N_SAMPLES}"
echo "Seed       : ${SEED}"
echo "Output dir : ${OUTPUT_DIR}"
echo "CPUs       : ${SLURM_CPUS_PER_TASK}"
echo "Started at : $(date +"%T")"
echo "=================================================="

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

echo "Python : $(which python)"
echo "LaTeX  : $(which latex || echo 'latex not found')"

mkdir -p "${OUTPUT_DIR}"

papermill GPR.ipynb "${OUTPUT_DIR}/GPR_residual_output.ipynb" \
    -p OUTPUT_FOLDER        "${OUTPUT_DIR}"  \
    -p SEED                 ${SEED}          \
    -p EXPERIMENT_MAX_SAMPLES ${N_SAMPLES}   \
    -p RESTART_OPTIMIZER    10               \
    -p STRAT_BINS_SAMPLING  10               \
    -p STRAT_BINS_SPLIT     2                \
    -p RUN_CV               True            \
    -p CV_FOLDS             5

echo "Finished at: $(date +"%T")"
