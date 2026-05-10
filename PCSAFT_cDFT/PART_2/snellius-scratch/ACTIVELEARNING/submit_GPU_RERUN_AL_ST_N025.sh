#!/bin/bash

#SBATCH -J AL_ST_N025_RERUN
#SBATCH -t 02:00:00
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /scratch-shared/draju/PART_2/ACTIVELEARNING/slurm-rerun-al-st-n025-%j.out

# Reruns AL-ST N025 Bubble and Gamma, which failed in the original run due to a
# ZMQ port conflict (kernel died before executing any cell). Jobs are run
# SEQUENTIALLY to avoid port conflicts from concurrent papermill kernels.

set -euo pipefail

echo "Job started at: $(date +"%T")"

module load 2025
module load MPICH/4.3.0-GCC-14.2.0-CUDA-12.8.0
module load Python/3.13.1-GCCcore-14.2.0

export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/mps_pipe
export CUDA_MPS_LOG_DIRECTORY=$TMPDIR/mps_log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

cleanup() {
    echo "Cleaning up at $(date +"%T")"
    echo quit | nvidia-cuda-mps-control 2>/dev/null
}
trap cleanup EXIT

cd /scratch-shared/draju/PART_2/ACTIVELEARNING
source /gpfs/home6/draju/A6/.A6/bin/activate

echo "CUDA Version: ${EBVERSIONCUDA}"

DATA_ROOT="/scratch-shared/draju/PART_2/ACTIVELEARNING/COMBINED"
OUTPUT_ROOT="/scratch-shared/draju/PART_2/ACTIVELEARNING/OUTPUTS"
SEED_BUBBLE=50015
SEED_GAMMA=52225

mkdir -p "${OUTPUT_ROOT}/AL_ST/N025/Bubble"
mkdir -p "${OUTPUT_ROOT}/AL_ST/N025/Gamma"

echo "=== [1/2] AL_ST N025 Bubble ==="
CUDA_VISIBLE_DEVICES=0 papermill TabPFNBubble_Calc.ipynb \
    "${OUTPUT_ROOT}/AL_ST/N025/TabPFNBubble_AL_ST_N025_output.ipynb" \
    -p PLOT_FOLDER "${OUTPUT_ROOT}/AL_ST/N025/Bubble" \
    -p TEST_ROWS   None \
    -p SEED        "${SEED_BUBBLE}" \
    -p DATA_PATH   "${DATA_ROOT}/AL_ST/N025.csv" \
    > "${OUTPUT_ROOT}/AL_ST/N025/TabPFNBubble_AL_ST_N025.log" 2>&1
echo "    Done at $(date +"%T")"

echo "=== [2/2] AL_ST N025 Gamma ==="
CUDA_VISIBLE_DEVICES=0 papermill TabPFNGamma_Calc.ipynb \
    "${OUTPUT_ROOT}/AL_ST/N025/TabPFNGamma_AL_ST_N025_output.ipynb" \
    -p PLOT_FOLDER "${OUTPUT_ROOT}/AL_ST/N025/Gamma" \
    -p TEST_ROWS   None \
    -p SEED        "${SEED_GAMMA}" \
    -p DATA_PATH   "${DATA_ROOT}/AL_ST/N025.csv" \
    > "${OUTPUT_ROOT}/AL_ST/N025/TabPFNGamma_AL_ST_N025.log" 2>&1
echo "    Done at $(date +"%T")"

echo "All reruns completed at: $(date +"%T")"
