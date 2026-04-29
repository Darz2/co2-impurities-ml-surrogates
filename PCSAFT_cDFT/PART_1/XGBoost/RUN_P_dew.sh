#!/bin/bash
#SBATCH --job-name=ML-XGB_Pdew
#SBATCH --partition=parallel-short
#SBATCH --time=6:00:00
#SBATCH --exclude=c171
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

OUTPUT_DIR="SLURM_Pdew"
mkdir -p "${OUTPUT_DIR}"

papermill XGBoost_P_dew.ipynb "${OUTPUT_DIR}/XGB_P_dew_output.ipynb" \
    -p OUTPUT_FOLDER "${OUTPUT_DIR}" \
    -p SEED 8005515

echo "Job finished at: $(date +"%T")"
