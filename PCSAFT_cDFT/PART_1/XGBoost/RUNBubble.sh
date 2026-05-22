#!/bin/bash
#SBATCH --job-name=ML-XGBoost_Pbubble
#SBATCH --partition=highmem
#SBATCH --time=24:00:00
#SBATCH --nodelist=c108
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

set -euo pipefail

start_time=$(date +"%T")
echo "Job started at: $start_time"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

echo "Python: $(which python)"
echo "LaTeX : $(which latex || echo 'latex not found')"

OUTPUT_DIR="SLURM_Pbubble"
mkdir -p "${OUTPUT_DIR}"

TEST_ROWS="${TEST_ROWS:-None}"

papermill XGBoostBubble.ipynb "${OUTPUT_DIR}/XGBoostBubble_output.ipynb" \
    -p OUTPUT_FOLDER "${OUTPUT_DIR}" \
    -p TEST_ROWS "${TEST_ROWS}" \
    -p SEED 50015

echo "Job finished at: $(date +"%T")"
