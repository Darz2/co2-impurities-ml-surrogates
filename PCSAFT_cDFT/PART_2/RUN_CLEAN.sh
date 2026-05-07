#!/bin/bash
#SBATCH --job-name=cDFT_clean
#SBATCH --partition=highmem
#SBATCH --nodelist=c109
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=4:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

# Force non-interactive Agg backend so workers don't try to open a display
export MPLBACKEND=Agg

# Disable BLAS/OpenMP threading — joblib controls parallelism here
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=== clean jobs started at $(date '+%F %T') on $(hostname) ==="
echo "CPUs: ${SLURM_CPUS_PER_TASK}"

N_EACH=$(( SLURM_CPUS_PER_TASK / 2 ))
echo "Running RANDOM and STRATIFIED in parallel — ${N_EACH} joblib workers each"

# RANDOM — background subshell with its own SLURM_CPUS_PER_TASK
(
    export SLURM_CPUS_PER_TASK=${N_EACH}
    cd "${SCRIPT_DIR}/RANDOM"
    echo "[RANDOM] start $(date '+%T')"
    papermill clean.ipynb clean_output.ipynb
    echo "[RANDOM] done  $(date '+%T')"
) &
PID_R=$!

# STRATIFIED — background subshell
(
    export SLURM_CPUS_PER_TASK=${N_EACH}
    cd "${SCRIPT_DIR}/STRATIFIED"
    echo "[STRATIFIED] start $(date '+%T')"
    papermill clean.ipynb clean_output.ipynb
    echo "[STRATIFIED] done  $(date '+%T')"
) &
PID_S=$!

wait $PID_R || { echo "ERROR: RANDOM/clean.ipynb failed"; exit 1; }
wait $PID_S || { echo "ERROR: STRATIFIED/clean.ipynb failed"; exit 1; }

echo ""
echo "=== All clean jobs finished at $(date '+%F %T') ==="
