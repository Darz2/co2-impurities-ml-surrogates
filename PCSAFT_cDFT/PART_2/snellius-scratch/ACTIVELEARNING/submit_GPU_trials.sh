#!/bin/bash

#SBATCH -J AL_Trials
#SBATCH -t 48:00:00
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /scratch-shared/draju/PART_2/ACTIVELEARNING/slurm-trials-%j.out

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
GPU_MON_PID=""

gpu_monitor() {
    local logfile=$1
    local interval=${2:-10}

    echo "timestamp,index,sm_pct,mem_pct,power_w,power_limit_w,temp_c,vram_used_mib,vram_total_mib,pid_count" > "$logfile"

    while true; do
        nvidia-smi \
            --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits > /tmp/gpu_stats.csv

        nvidia-smi \
            --query-compute-apps=gpu_uuid,pid \
            --format=csv,noheader,nounits > /tmp/gpu_procs.csv

        pid_counts=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | while read -r g; do
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

cleanup() {
    echo "Cleaning up at $(date +"%T")"
    [[ -n "$GPU_MON_PID" ]] && kill "$GPU_MON_PID" 2>/dev/null
    echo quit | nvidia-cuda-mps-control 2>/dev/null
}
trap cleanup EXIT

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
cd /scratch-shared/draju/PART_2/ACTIVELEARNING
source /gpfs/home6/draju/A6/.A6/bin/activate

echo "CUDA Version: ${EBVERSIONCUDA}"
echo "MPICH Version: ${EBVERSIONMPICH}"
echo "Python Version: ${EBVERSIONPYTHON}"

DATA_ROOT="/scratch-shared/draju/PART_2/ACTIVELEARNING/COMBINED"
OUTPUT_ROOT="/scratch-shared/draju/PART_2/ACTIVELEARNING/OUTPUTS"

TEST_ROWS="${TEST_ROWS:-None}"

SEED_BUBBLE=50015
SEED_GAMMA=52225

NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "GPUs available: ${NUM_GPUS}"

# Max concurrent trial-pairs running at once (Bubble+Gamma sequential per trial)
MAX_PARALLEL=${MAX_PARALLEL:-8}

AL_TYPES=(AL_ST AL_MT)
SIZES=(N025 N050 N075 N100)
N_TRIALS=20

# ─────────────────────────────────────────────────────────────────────────────
# run_trial: runs Bubble then Gamma sequentially for one trial
# ─────────────────────────────────────────────────────────────────────────────
run_trial() {
    local gpu_id="$1"
    local al_type="$2"
    local size="$3"
    local trial_id="$4"   # e.g. trial_00

    local data_path="${DATA_ROOT}/${al_type}/${size}/${trial_id}.csv"
    local out_dir="${OUTPUT_ROOT}/${al_type}/${size}/${trial_id}"
    mkdir -p "${out_dir}"

    local bubble_log="${out_dir}/TabPFNBubble_${al_type}_${size}_${trial_id}.log"
    local gamma_log="${out_dir}/TabPFNGamma_${al_type}_${size}_${trial_id}.log"

    echo "[GPU ${gpu_id}] START Bubble  ${al_type}/${size}/${trial_id}  $(date +"%T")"
    CUDA_VISIBLE_DEVICES="${gpu_id}" papermill TabPFNBubble_Calc.ipynb \
        "${out_dir}/TabPFNBubble_${al_type}_${size}_${trial_id}_output.ipynb" \
        -p PLOT_FOLDER "${out_dir}/Bubble" \
        -p TEST_ROWS   "${TEST_ROWS}" \
        -p SEED        "${SEED_BUBBLE}" \
        -p DATA_PATH   "${data_path}" \
        > "${bubble_log}" 2>&1
    echo "[GPU ${gpu_id}] DONE  Bubble  ${al_type}/${size}/${trial_id}  $(date +"%T")"

    echo "[GPU ${gpu_id}] START Gamma   ${al_type}/${size}/${trial_id}  $(date +"%T")"
    CUDA_VISIBLE_DEVICES="${gpu_id}" papermill TabPFNGamma_Calc.ipynb \
        "${out_dir}/TabPFNGamma_${al_type}_${size}_${trial_id}_output.ipynb" \
        -p PLOT_FOLDER "${out_dir}/Gamma" \
        -p TEST_ROWS   "${TEST_ROWS}" \
        -p SEED        "${SEED_GAMMA}" \
        -p DATA_PATH   "${data_path}" \
        > "${gamma_log}" 2>&1
    echo "[GPU ${gpu_id}] DONE  Gamma   ${al_type}/${size}/${trial_id}  $(date +"%T")"
}

# ─────────────────────────────────────────────────────────────────────────────
# Start GPU monitor
# ─────────────────────────────────────────────────────────────────────────────
gpu_monitor "/scratch-shared/draju/PART_2/ACTIVELEARNING/gpu_monitor_trials.csv" 30

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch all trials with bounded parallelism (semaphore via active PID count)
# ─────────────────────────────────────────────────────────────────────────────
PID_LIST=()
idx=0

for al_type in "${AL_TYPES[@]}"; do
    for size in "${SIZES[@]}"; do
        for trial_num in $(seq 0 $(( N_TRIALS - 1 ))); do
            trial_id=$(printf "trial_%02d" "${trial_num}")
            gpu_id=$(( idx % NUM_GPUS ))

            # Wait if we have reached MAX_PARALLEL active jobs
            while (( ${#PID_LIST[@]} >= MAX_PARALLEL )); do
                # Reap any finished processes
                new_list=()
                for pid in "${PID_LIST[@]}"; do
                    if kill -0 "$pid" 2>/dev/null; then
                        new_list+=("$pid")
                    else
                        wait "$pid" || true
                    fi
                done
                PID_LIST=("${new_list[@]+"${new_list[@]}"}")
                (( ${#PID_LIST[@]} >= MAX_PARALLEL )) && sleep 5
            done

            run_trial "${gpu_id}" "${al_type}" "${size}" "${trial_id}" &
            PID_LIST+=($!)
            idx=$(( idx + 1 ))
        done
    done
done

# Wait for all remaining jobs
wait "${PID_LIST[@]+"${PID_LIST[@]}"}"

echo "All trials completed at: $(date +"%T")"
