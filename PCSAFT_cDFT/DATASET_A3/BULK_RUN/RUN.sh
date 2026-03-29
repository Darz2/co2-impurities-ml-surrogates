#!/bin/bash
#SBATCH --job-name=cDFT_SEC_KIJeq0
#SBATCH --partition=serial
#SBATCH --time=7-00:00:00
#SBATCH --exclude=c171
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-99

set -euo pipefail

echo "Job started at: $(date '+%F %T')"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Error: This script must be submitted as a Slurm array job."
    echo "Submit with: sbatch $0"
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID}"
TASK_JOB_ID="${SLURM_JOB_ID}"

echo "Array job ID : ${ARRAY_JOB_ID}"
echo "Task job ID  : ${TASK_JOB_ID}"
echo "Task ID      : ${TASK_ID}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ ! -f "VLE_cDFT_SEC.ipynb" ]; then
    echo "Error: VLE_cDFT_SEC.ipynb not found in $(pwd)"
    exit 1
fi

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

command -v python >/dev/null 2>&1 || { echo "Error: python not found"; exit 1; }
command -v papermill >/dev/null 2>&1 || { echo "Error: papermill not found"; exit 1; }

echo "Python    : $(which python)"
echo "LaTeX     : $(which latex || echo 'latex not found')"
echo "Papermill : $(which papermill)"

OUTPUT_DIR="${ARRAY_JOB_ID}/${ARRAY_JOB_ID}_${TASK_ID}"
OUT_DIR="${OUTPUT_DIR}/OUTPUT"

mkdir -p "${OUT_DIR}"

papermill "VLE_cDFT_SEC.ipynb" "${OUTPUT_DIR}/VLE_cDFT_SEC_${TASK_ID}.ipynb" \
    -p FEED_INDEX "${TASK_ID}" \
    -p verbose False \
    -p OUTPUT_FOLDER "${OUT_DIR}" \
    -p N_T 30 \
    -p N_P 30

echo "Job finished at: $(date '+%F %T')"