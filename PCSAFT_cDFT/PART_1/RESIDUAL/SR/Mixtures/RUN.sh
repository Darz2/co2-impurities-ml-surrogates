#!/bin/bash
#SBATCH --job-name=SR_VLE_cDFT_SEC
#SBATCH --partition=serial
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

set -euo pipefail

echo "Job started at: $(date +"%T")"
echo "Running on node: $(hostname)"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

echo "Python:    $(which python)"
echo "LaTeX:     $(which latex 2>/dev/null || echo 'not found')"
echo "Papermill: $(which papermill)"

JOB_ID="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${JOB_ID}_SR_VLE_cDFT_SEC"
CSV_DIR="${OUTPUT_DIR}/CSV"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CSV_DIR}"

papermill VLE_cDFT_SEC.ipynb "${OUTPUT_DIR}/VLE_cDFT_SEC_${JOB_ID}.ipynb"

echo "Job finished at: $(date +"%T")"
