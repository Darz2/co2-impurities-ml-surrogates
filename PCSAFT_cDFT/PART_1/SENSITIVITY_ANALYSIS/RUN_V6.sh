#!/bin/bash
#SBATCH --job-name=cDFT_SENS_V6
#SBATCH --partition=highmem
#SBATCH --time=7-00:00:00
#SBATCH --nodelist=c109
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G

set -euo pipefail

echo "Job started at: $(date '+%F %T')"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

NOTEBOOK="VLE_IFT_V6_branch.ipynb"

# Branch selector for V6 (single P per T): "bubble" or "dew".
# Override at submit time, e.g.:   BRANCH=dew sbatch RUN_V6.sh
BRANCH="${BRANCH:-bubble}"
if [ "${BRANCH}" != "bubble" ] && [ "${BRANCH}" != "dew" ]; then
    echo "Error: BRANCH must be 'bubble' or 'dew', got '${BRANCH}'"
    exit 1
fi

if [ ! -f "${NOTEBOOK}" ]; then
    echo "Error: ${NOTEBOOK} not found in $(pwd)"
    exit 1
fi

COMPOSITIONS_CSV="CSV_feeds/Combined_compositions.csv"
if [ ! -f "${COMPOSITIONS_CSV}" ]; then
    echo "Error: ${COMPOSITIONS_CSV} not found in $(pwd)"
    exit 1
fi

N_COMPOSITIONS="$(python3 -c "import csv; print(sum(1 for _ in open('${COMPOSITIONS_CSV}', newline='')) - 1)")"
if [ "${N_COMPOSITIONS}" -le 0 ]; then
    echo "Error: no compositions found in ${COMPOSITIONS_CSV}"
    exit 1
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    LAST_INDEX=$((N_COMPOSITIONS - 1))
    echo "Submitting ${N_COMPOSITIONS} compositions as array 0-${LAST_INDEX}  (BRANCH=${BRANCH})"
    # Re-export BRANCH so the array tasks inherit the same selection.
    sbatch --export=ALL,BRANCH="${BRANCH}" --array="0-${LAST_INDEX}" "$0"
    exit 0
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID}"
TASK_JOB_ID="${SLURM_JOB_ID}"

echo "Array job ID : ${ARRAY_JOB_ID}"
echo "Task job ID  : ${TASK_JOB_ID}"
echo "Task ID      : ${TASK_ID}"
echo "Branch       : ${BRANCH}"

if [ "${TASK_ID}" -ge "${N_COMPOSITIONS}" ]; then
    echo "Error: Task ID ${TASK_ID} is out of range for ${N_COMPOSITIONS} compositions."
    echo "Valid task IDs: 0-$((N_COMPOSITIONS - 1))"
    exit 1
fi
echo "Composition count: ${N_COMPOSITIONS}"

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

# V6 parallelism strategy (same as V5):
#   - notebook ThreadPool uses NUM_THREADS = SLURM_CPUS_PER_TASK workers
#   - each worker runs ONE cDFT solve; BLAS/Rayon stay single-threaded to
#     avoid 48x48 oversubscription on c109.
# V6 sweeps ONE P per T (vs V5's cDFT_NP=10), so wall time is ~1/10 of V5.
CPUS="${SLURM_CPUS_PER_TASK:-1}"
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
echo "Outer workers (NUM_THREADS) : ${CPUS}"
echo "Inner BLAS/Rayon threads    : 1  (avoid oversubscription)"

command -v python >/dev/null 2>&1 || { echo "Error: python not found"; exit 1; }
command -v papermill >/dev/null 2>&1 || { echo "Error: papermill not found"; exit 1; }

echo "Python    : $(which python)"
echo "LaTeX     : $(which latex || echo 'latex not found')"
echo "Papermill : $(which papermill)"

# Output dir embeds the branch so bubble & dew runs don't collide.
OUTPUT_DIR="${ARRAY_JOB_ID}_${BRANCH}/${ARRAY_JOB_ID}_${TASK_ID}_${BRANCH}"
CSV_DIR="${OUTPUT_DIR}/CSV"
PLOT_DIR="${OUTPUT_DIR}/PLOTS"

mkdir -p "${CSV_DIR}" "${PLOT_DIR}"

papermill "${NOTEBOOK}" "${OUTPUT_DIR}/VLE_IFT_V6_${BRANCH}_${TASK_ID}.ipynb" \
    -p FEED_INDEX "${TASK_ID}" \
    -p SLURM_RUN True \
    -p verbose False \
    -p CSV_FOLDER "${CSV_DIR}" \
    -p PLOT_FOLDER "${PLOT_DIR}" \
    -p CRITICAL_REGION_ENHANCEMENTS False \
    -p BRANCH "${BRANCH}" \
    -p NUM_THREADS "${CPUS}"

echo "Job finished at: $(date '+%F %T')"
