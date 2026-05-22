#!/bin/bash

#SBATCH -J GPU_TabPFN
#SBATCH -t 04:00:00
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:4

set -euo pipefail

start_time=$(date +"%T")
echo "Job started at: $start_time"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load 2025
module load MPICH/4.3.0-GCC-14.2.0-CUDA-12.8.0
module load Python/3.13.1-GCCcore-14.2.0

cd /gpfs/home6/draju/A6/TabPFN/TEST
source /gpfs/home6/draju/A6/.A6/bin/activate

echo "CUDA Version: ${EBVERSIONCUDA}"
echo "MPICH Version: ${EBVERSIONMPICH}"
echo "Python Version: ${EBVERSIONPYTHON}"
echo "LaTeX : $(which latex || echo 'latex not found')"


OUTPUT_DIR="/gpfs/home6/draju/A6/TabPFN/TEST/SLURM_Pbubble"
mkdir -p "${OUTPUT_DIR}"

TEST_ROWS="${TEST_ROWS:-None}"

# python3 torch_test.py

papermill TabPFNBubble_GPU.ipynb "${OUTPUT_DIR}/TabPFNBubble_GPU_output.ipynb" \
    -p PLOT_FOLDER "${OUTPUT_DIR}" \
    -p SLURM_RUN_FULL True \
    -p TEST_ROWS "${TEST_ROWS}" \
    -p SEED 454015

echo "Job finished at: $(date +"%T")"
