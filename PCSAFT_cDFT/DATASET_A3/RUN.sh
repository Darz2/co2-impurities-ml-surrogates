#!/bin/bash
#SBATCH --job-name=cDFT_SEC_IFT
#SBATCH --partition=serial
#SBATCH --time=7-00:00:00
#SBATCH --exclude=c171
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --array=16

set -euo pipefail

start_time=$(date +"%T")
echo "Job started at: $start_time"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Error: This script must be submitted as a Slurm array job."
    echo "Submit with: sbatch $0"
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"
JOB_ID="${SLURM_ARRAY_JOB_ID}"

echo "TASK_ID: ${TASK_ID}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

echo "Python: $(which python)"
echo "LaTeX : $(which latex || echo 'latex not found')"
echo "Papermill: $(which papermill)"

OUTPUT_DIR="${JOB_ID}_${TASK_ID}"
CSV_DIR="${OUTPUT_DIR}/OUTPUT"

mkdir -p "${CSV_DIR}"

papermill VLE_cDFT_SEC.ipynb "${OUTPUT_DIR}/VLE_cDFT_SEC_${TASK_ID}.ipynb" \
    -p SLURM_RUN True \
    -p FEED_INDEX "${TASK_ID}" \
    -p verbose False \
    -p CSV_FOLDER "${CSV_DIR}"

echo "Job finished at: $(date +"%T")"