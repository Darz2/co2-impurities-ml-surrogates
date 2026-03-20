#!/bin/bash
#SBATCH --job-name=TabPFN_Gamma
#SBATCH --partition=highmem
#SBATCH --time=1-00:00:00
#SBATCH --exclude=c171
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

set -euo pipefail

start_time=$(date +"%T")
echo "Job started at: $start_time"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"

echo "Python: $(which python)"
echo "LaTeX : $(which latex || echo 'latex not found')"

OUTPUT_DIR="outputs_TabPFN_Gamma"
mkdir -p "${OUTPUT_DIR}"

papermill TabPFN.ipynb "${OUTPUT_DIR}/TabPFN_Gamma_output.ipynb"

echo "Job finished at: $(date +"%T")"