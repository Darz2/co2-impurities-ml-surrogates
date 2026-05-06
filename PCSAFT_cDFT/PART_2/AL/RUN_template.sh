#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Template SLURM script for AL-selected feed batches.
# Copy this file and VLE_IFT_V4.ipynb into the target dataset directory,
# then update --array to match the number of selected feeds (N-1).
#
# Steps:
#   1. Run 01_generate_pool.ipynb  → AL_POOL/composition_pool.csv
#   2. Run 02_GPR_AL.ipynb         → AL_OUTPUTS/Combined_compositions.csv
#   3. mkdir -p ../DATASET_A4/CSV_feeds
#      cp AL_OUTPUTS/Combined_compositions.csv ../DATASET_A4/CSV_feeds/
#      cp AL_POOL/KIJ_pairs.json               ../DATASET_A4/CSV_feeds/
#      cp ../DATASET_A1/VLE_IFT_V4.ipynb       ../DATASET_A4/
#      cp RUN_template.sh                       ../DATASET_A4/RUN.sh
#   4. Edit --array below to 0-<N_SELECTED-1>
#   5. cd ../DATASET_A4 && sbatch RUN.sh
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=cDFT_AL
#SBATCH --partition=serial
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-99          # <── update to 0-<N_SELECTED-1>

set -euo pipefail

echo "Job started at: $(date '+%F %T')"

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Error: submit as a SLURM array job: sbatch $0"
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID}"

echo "Array job ID : ${ARRAY_JOB_ID}"
echo "Task ID      : ${TASK_ID}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

OUTPUT_DIR="${ARRAY_JOB_ID}/${ARRAY_JOB_ID}_${TASK_ID}"
CSV_DIR="${OUTPUT_DIR}/CSV"
PLOT_DIR="${OUTPUT_DIR}/PLOTS"
mkdir -p "${CSV_DIR}" "${PLOT_DIR}"

papermill "VLE_IFT_V4.ipynb" "${OUTPUT_DIR}/VLE_IFT_V4_${TASK_ID}.ipynb" \
    -p FEED_INDEX "${TASK_ID}" \
    -p SLURM_RUN  True \
    -p verbose    False \
    -p CSV_FOLDER "${CSV_DIR}" \
    -p PLOT_FOLDER "${PLOT_DIR}"

echo "Job finished at: $(date '+%F %T')"
