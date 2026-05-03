#!/bin/bash
#SBATCH --job-name=HPO-TabPFN_Pbubble
#SBATCH --partition=highmem
#SBATCH --time=7-00:00:00
#SBATCH --nodelist=c109
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G

set -euo pipefail

start_time=$(date +"%T")
echo "Job started at: $start_time"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

echo "Python: $(which python)"
echo "LaTeX : $(which latex || echo 'latex not found')"

OUTPUT_DIR="SLURM_Pbubble"
mkdir -p "${OUTPUT_DIR}"

TEST_ROWS="${TEST_ROWS:-None}"
N_TRIALS=100

echo "N_TRIALS=${N_TRIALS}"

papermill TabPFNBubble_Calc.ipynb "${OUTPUT_DIR}/TabPFNBubble_Calc_output.ipynb" \
    -p PLOT_FOLDER "${OUTPUT_DIR}" \
    -p TEST_ROWS "${TEST_ROWS}" \
    -p N_TRIALS "${N_TRIALS}" \
    -p SEED 454015

echo "Job finished at: $(date +"%T")"
