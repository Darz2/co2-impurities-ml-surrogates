#!/bin/bash
#SBATCH --job-name=cDFT_TIMING_V5
#SBATCH --partition=highmem
#SBATCH --time=12:00:00
#SBATCH --nodelist=c109
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G

# ---------------------------------------------------------------------------
# Re-time the V5 cDFT contour-map sweep for the 5 study feeds, recording the
# phase-equilibria stage and the cDFT interfacial sweep SEPARATELY per feed.
#
#   * Same physics as SENSITIVITY_ANALYSIS/VLE_IFT_V5.ipynb (byte-for-byte;
#     only timing markers + a writer cell were added -> VLE_IFT_V5_TIMED.ipynb).
#   * DEFAULT grid (SLURM_RUN=False -> ngrid=500, lgrid=100, T_STEP=5,
#     cDFT_NP=5). No critical-region enhancement.
#   * HYBRID threading: OUTER Python ThreadPool workers x INNER rayon/BLAS
#     threads, with OUTER*INNER = cpus-per-task. Override at submit time:
#       OUTER=4 INNER=4 sbatch RUN_TIMING_V5.sh
#
# Submit (5 feeds as an array 0-4):   sbatch RUN_TIMING_V5.sh
# ---------------------------------------------------------------------------

set -euo pipefail
echo "Job started at: $(date '+%F %T')"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

NOTEBOOK="VLE_IFT_V5_TIMED.ipynb"
COMPOSITIONS_CSV="CSV_feeds/Combined_compositions.csv"

[ -f "${NOTEBOOK}" ]        || { echo "Error: ${NOTEBOOK} not found in $(pwd)"; exit 1; }
[ -f "${COMPOSITIONS_CSV}" ] || { echo "Error: ${COMPOSITIONS_CSV} not found in $(pwd)"; exit 1; }

N_COMPOSITIONS="$(python3 -c "import csv; print(sum(1 for _ in open('${COMPOSITIONS_CSV}', newline='')) - 1)")"
[ "${N_COMPOSITIONS}" -gt 0 ] || { echo "Error: no compositions in ${COMPOSITIONS_CSV}"; exit 1; }

# Resubmit self as an array over all feeds if not already an array task.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    LAST_INDEX=$((N_COMPOSITIONS - 1))
    echo "Submitting ${N_COMPOSITIONS} feeds as array 0-${LAST_INDEX}"
    sbatch --export=ALL,OUTER="${OUTER:-}",INNER="${INNER:-}" --array="0-${LAST_INDEX}" "$0"
    exit 0
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID}"
echo "Array job ID : ${ARRAY_JOB_ID}"
echo "Task ID      : ${TASK_ID}"

if [ "${TASK_ID}" -ge "${N_COMPOSITIONS}" ]; then
    echo "Error: Task ID ${TASK_ID} out of range 0-$((N_COMPOSITIONS - 1))"; exit 1
fi

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"
export MPLBACKEND=Agg

# --- HYBRID threading: OUTER workers x INNER rayon/BLAS threads = CPUS ---------
CPUS="${SLURM_CPUS_PER_TASK:-16}"
OUTER="${OUTER:-4}"
INNER="${INNER:-$(( CPUS / OUTER ))}"
[ "${INNER}" -ge 1 ] || INNER=1
PRODUCT=$(( OUTER * INNER ))
if [ "${PRODUCT}" -gt "${CPUS}" ]; then
    echo "WARNING: OUTER(${OUTER}) x INNER(${INNER}) = ${PRODUCT} > CPUS(${CPUS}) -> oversubscription"
fi

export RAYON_NUM_THREADS="${INNER}"
export OMP_NUM_THREADS="${INNER}"
export MKL_NUM_THREADS="${INNER}"
export OPENBLAS_NUM_THREADS="${INNER}"
export NUMEXPR_NUM_THREADS="${INNER}"
echo "HYBRID threads: OUTER pool = ${OUTER} workers | INNER rayon/BLAS = ${INNER} | CPUS = ${CPUS}"

command -v papermill >/dev/null 2>&1 || { echo "Error: papermill not found"; exit 1; }
echo "Python    : $(which python)"
echo "Papermill : $(which papermill)"

OUTPUT_DIR="RESULTS/${ARRAY_JOB_ID}_${TASK_ID}"
CSV_DIR="${OUTPUT_DIR}/CSV"
PLOT_DIR="${OUTPUT_DIR}/PLOTS"
mkdir -p "${CSV_DIR}" "${PLOT_DIR}" "RESULTS/timing"

TIMING_JSON="RESULTS/timing/timing_feed_${TASK_ID}.json"

papermill "${NOTEBOOK}" "${OUTPUT_DIR}/VLE_IFT_V5_TIMED_${TASK_ID}.ipynb" \
    -p FEED_INDEX "${TASK_ID}" \
    -p SLURM_RUN False \
    -p verbose False \
    -p CSV_FOLDER "${CSV_DIR}" \
    -p PLOT_FOLDER "${PLOT_DIR}" \
    -p CRITICAL_REGION_ENHANCEMENTS False \
    -p IFT_CONTOUR_PLOT_ENHANCEMENTS False \
    -p NUM_THREADS "${OUTER}" \
    -p TIMING_FILE "${TIMING_JSON}"

echo "Per-feed timing written to ${TIMING_JSON}"
echo "Job finished at: $(date '+%F %T')"
