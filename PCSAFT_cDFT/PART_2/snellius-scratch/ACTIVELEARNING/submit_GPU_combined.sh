#!/bin/bash

#SBATCH -J ActiveLearning
#SBATCH -t 08:00:00
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
##SBATCH --mem=720G
#SBATCH --gres=gpu:1
#SBATCH -o /scratch-shared/draju/PART_2/ACTIVELEARNING/slurm-combined-al-%j.out

set -euo pipefail

start_time=$(date +"%T")
echo "Job started at: ${start_time}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load 2025
module load MPICH/4.3.0-GCC-14.2.0-CUDA-12.8.0
module load Python/3.13.1-GCCcore-14.2.0

# ============================================================
# MPS SETUP
# ============================================================
export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/mps_pipe
export CUDA_MPS_LOG_DIRECTORY=$TMPDIR/mps_log

mkdir -p "$CUDA_MPS_PIPE_DIRECTORY"
mkdir -p "$CUDA_MPS_LOG_DIRECTORY"

nvidia-cuda-mps-control -d

# ─────────────────────────────────────────────────────────────────────────────
# GPU monitor
# ─────────────────────────────────────────────────────────────────────────────
gpu_monitor() {
    local logfile=$1
    local interval=${2:-5}

    echo "timestamp,index,sm_pct,mem_pct,power_w,power_limit_w,temp_c,vram_used_mib,vram_total_mib,pid_count" > "$logfile"

    while true; do
        nvidia-smi \
            --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits > /tmp/gpu_stats.csv

        nvidia-smi \
            --query-compute-apps=gpu_uuid,pid \
            --format=csv,noheader,nounits > /tmp/gpu_procs.csv

        pid_counts=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | while read g; do
            count=$(wc -l < /tmp/gpu_procs.csv)
            echo "$count"
        done | tr '\n' ',' | sed 's/,$//')

        paste -d',' \
            /tmp/gpu_stats.csv \
            <(echo "$pid_counts") >> "$logfile"

        sleep "$interval"
    done &
    GPU_MON_PID=$!
}

stop_monitor() {
    kill "$1" 2>/dev/null
}

cleanup() {
    echo "Cleaning up..."
    stop_monitor "$GPU_MON_PID"
    echo quit | nvidia-cuda-mps-control 2>/dev/null
}

## SIMULATIONS START HERE
gpu_monitor "/scratch-shared/draju/PART_2/ACTIVELEARNING/gpu_monitor.csv" 10

# Kill monitor and MPS on job exit (normal or error)
trap cleanup EXIT

cd /scratch-shared/draju/PART_2/ACTIVELEARNING
source /gpfs/home6/draju/A6/.A6/bin/activate

echo "CUDA Version: ${EBVERSIONCUDA}"
echo "MPICH Version: ${EBVERSIONMPICH}"
echo "Python Version: ${EBVERSIONPYTHON}"

DATA_ROOT="/scratch-shared/draju/PART_2/ACTIVELEARNING/COMBINED"
OUTPUT_ROOT="/scratch-shared/draju/PART_2/ACTIVELEARNING/OUTPUTS"

TEST_ROWS="${TEST_ROWS:-None}"

# Seeds (same as original)
SEED_BUBBLE=50015
SEED_GAMMA=52225

run_size() {
    local gpu_id="$1"
    local al_type="$2"   # AL_ST or AL_MT
    local size="$3"      # N025, N050, N075, N100

    local data_path="${DATA_ROOT}/${al_type}/${size}.csv"
    local out_dir="${OUTPUT_ROOT}/${al_type}/${size}"
    mkdir -p "${out_dir}"

    local bubble_log="${out_dir}/TabPFNBubble_${al_type}_${size}.log"
    local gamma_log="${out_dir}/TabPFNGamma_${al_type}_${size}.log"

    echo "[GPU ${gpu_id}] Starting ${al_type}/${size} Bubble at $(date +"%T") -> ${bubble_log}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" papermill TabPFNBubble_Calc.ipynb \
        "${out_dir}/TabPFNBubble_${al_type}_${size}_output.ipynb" \
        -p PLOT_FOLDER "${out_dir}/Bubble" \
        -p TEST_ROWS  "${TEST_ROWS}" \
        -p SEED       "${SEED_BUBBLE}" \
        -p DATA_PATH  "${data_path}" \
        > "${bubble_log}" 2>&1
    echo "[GPU ${gpu_id}] Finished ${al_type}/${size} Bubble at $(date +"%T")"

    echo "[GPU ${gpu_id}] Starting ${al_type}/${size} Gamma at $(date +"%T") -> ${gamma_log}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" papermill TabPFNGamma_Calc.ipynb \
        "${out_dir}/TabPFNGamma_${al_type}_${size}_output.ipynb" \
        -p PLOT_FOLDER "${out_dir}/Gamma" \
        -p TEST_ROWS  "${TEST_ROWS}" \
        -p SEED       "${SEED_GAMMA}" \
        -p DATA_PATH  "${data_path}" \
        > "${gamma_log}" 2>&1
    echo "[GPU ${gpu_id}] Finished ${al_type}/${size} Gamma at $(date +"%T")"
}

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "GPUs available: ${NUM_GPUS}"

AL_TYPES=(AL_ST AL_MT)
SIZES=(N025 N050 N075 N100)
PID_LIST=()
idx=0
for al_type in "${AL_TYPES[@]}"; do
    for size in "${SIZES[@]}"; do
        gpu_id=$(( idx % NUM_GPUS ))
        run_size "${gpu_id}" "${al_type}" "${size}" &
        PID_LIST+=($!)
        idx=$(( idx + 1 ))
    done
done

wait "${PID_LIST[@]}"

cleanup  # Ensure monitor and MPS are stopped

echo "All sizes completed at: $(date +"%T")"
