#!/bin/bash
#SBATCH --job-name=cDFT_V4_trial
#SBATCH --partition=serial
#SBATCH --time=7-00:00:00
#SBATCH --exclude=c171
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

set -euo pipefail

SIZE="${1:?Usage: sbatch $0 SIZE TRIAL  (e.g. N025 trial_00)}"
TRIAL="${2:?Usage: sbatch $0 SIZE TRIAL  (e.g. N025 trial_00)}"

echo "Job started at: $(date '+%F %T')"
echo "Size: ${SIZE}  Trial: ${TRIAL}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ ! -f "VLE_IFT_V4.ipynb" ]; then
    echo "Error: VLE_IFT_V4.ipynb not found in $(pwd)"
    exit 1
fi

COMPOSITIONS_CSV="OUTPUT/${SIZE}/${TRIAL}.csv"
if [ ! -f "${COMPOSITIONS_CSV}" ]; then
    echo "Error: ${COMPOSITIONS_CSV} not found in $(pwd)"
    exit 1
fi

N_COMPOSITIONS="$(python3 -c "import csv; print(sum(1 for _ in open('${COMPOSITIONS_CSV}', newline='')) - 1)")"
if [ "${N_COMPOSITIONS}" -le 0 ]; then
    echo "Error: no compositions found in ${COMPOSITIONS_CSV}"
    exit 1
fi

# First call: not inside an array yet — self-submit as an array job
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    LAST_INDEX=$((N_COMPOSITIONS - 1))
    echo "Submitting ${N_COMPOSITIONS} compositions as array 0-${LAST_INDEX}"
    sbatch --array="0-${LAST_INDEX}" "$0" "${SIZE}" "${TRIAL}"
    exit 0
fi

# Array task: run one composition
TASK_ID="${SLURM_ARRAY_TASK_ID}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID}"

echo "Array job ID : ${ARRAY_JOB_ID}"
echo "Task job ID  : ${SLURM_JOB_ID}"
echo "Task ID      : ${TASK_ID} / $((N_COMPOSITIONS - 1))"

if [ "${TASK_ID}" -ge "${N_COMPOSITIONS}" ]; then
    echo "Error: Task ID ${TASK_ID} out of range for ${N_COMPOSITIONS} compositions."
    exit 1
fi

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

command -v python    >/dev/null 2>&1 || { echo "Error: python not found";    exit 1; }
command -v papermill >/dev/null 2>&1 || { echo "Error: papermill not found"; exit 1; }

echo "Python    : $(which python)"
echo "LaTeX     : $(which latex 2>/dev/null || echo 'not found')"
echo "Papermill : $(which papermill)"

OUTPUT_DIR="RESULTS/${SIZE}/${TRIAL}/${ARRAY_JOB_ID}_${TASK_ID}"
CSV_DIR="${OUTPUT_DIR}/CSV"
PLOT_DIR="${OUTPUT_DIR}/PLOTS"

mkdir -p "${CSV_DIR}" "${PLOT_DIR}"

papermill "VLE_IFT_V4.ipynb" "${OUTPUT_DIR}/VLE_IFT_V4_${TASK_ID}.ipynb" \
    -p FEED_INDEX            "${TASK_ID}" \
    -p COMPOSITIONS_CSV      "${COMPOSITIONS_CSV}" \
    -p SLURM_RUN             True \
    -p verbose               False \
    -p CSV_FOLDER            "${CSV_DIR}" \
    -p PLOT_FOLDER           "${PLOT_DIR}" \
    -p CRITICAL_REGION_ENHANCEMENTS False

echo "Job finished at: $(date '+%F %T')"
