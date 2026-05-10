#!/bin/bash

#SBATCH -J RANDOM_ALL_TRIALS
#SBATCH -t 04:00:00
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH -o /scratch-shared/draju/PART_2/RANDOM/slurm-all-trials-%j.out

# Runs all 20 trials × 4 sizes × Bubble + Gamma (160 notebooks total).
# 4 GPU workers run in parallel, each handling 5 trials sequentially on its
# own GPU — no MPS needed, no OOM risk within a GPU.
# Already-completed notebooks (output file > 50 KB) are skipped.

set -uo pipefail

echo "Job started at: $(date +"%T")"

module load 2025
module load MPICH/4.3.0-GCC-14.2.0-CUDA-12.8.0
module load Python/3.13.1-GCCcore-14.2.0

cd /scratch-shared/draju/PART_2/RANDOM
source /gpfs/home6/draju/A6/.A6/bin/activate

echo "CUDA Version: ${EBVERSIONCUDA}"

DATA_ROOT="/scratch-shared/draju/PART_2/RANDOM/COMBINED"
OUTPUT_ROOT="/scratch-shared/draju/PART_2/RANDOM/OUTPUTS"
SEED_BUBBLE=50015
SEED_GAMMA=50005

SIZES=(N025 N050 N075 N100)

# ── Worker: runs a list of trials on a given GPU, sequentially ────────────────
run_gpu_worker() {
    local gpu_id="$1"; shift
    local trials=("$@")

    for trial in "${trials[@]}"; do
        for size in "${SIZES[@]}"; do
            local out_dir="${OUTPUT_ROOT}/${size}"
            mkdir -p "${out_dir}/Bubble" "${out_dir}/Gamma"

            # Bubble
            local nb_bubble="${out_dir}/TabPFNBubble_${size}_${trial}_output.ipynb"
            if [[ -f "$nb_bubble" ]] && [[ $(stat -c%s "$nb_bubble") -gt 50000 ]]; then
                echo "[GPU${gpu_id}] SKIP  ${size}/${trial}/Bubble"
            else
                echo "[GPU${gpu_id}] START ${size}/${trial}/Bubble — $(date +"%T")"
                CUDA_VISIBLE_DEVICES="${gpu_id}" papermill TabPFNBubble_Calc.ipynb \
                    "$nb_bubble" \
                    -p PLOT_FOLDER "${out_dir}/Bubble" \
                    -p TEST_ROWS   None \
                    -p SEED        "${SEED_BUBBLE}" \
                    -p DATA_PATH   "${DATA_ROOT}/${size}/${trial}.csv" \
                    > "${out_dir}/TabPFNBubble_${size}_${trial}.log" 2>&1 \
                    || echo "[GPU${gpu_id}] ERROR ${size}/${trial}/Bubble — see log"
                echo "[GPU${gpu_id}] DONE  ${size}/${trial}/Bubble — $(date +"%T")"
            fi

            # Gamma
            local nb_gamma="${out_dir}/TabPFNGamma_${size}_${trial}_output.ipynb"
            if [[ -f "$nb_gamma" ]] && [[ $(stat -c%s "$nb_gamma") -gt 50000 ]]; then
                echo "[GPU${gpu_id}] SKIP  ${size}/${trial}/Gamma"
            else
                echo "[GPU${gpu_id}] START ${size}/${trial}/Gamma  — $(date +"%T")"
                CUDA_VISIBLE_DEVICES="${gpu_id}" papermill TabPFNGamma_Calc.ipynb \
                    "$nb_gamma" \
                    -p PLOT_FOLDER "${out_dir}/Gamma" \
                    -p TEST_ROWS   None \
                    -p SEED        "${SEED_GAMMA}" \
                    -p DATA_PATH   "${DATA_ROOT}/${size}/${trial}.csv" \
                    > "${out_dir}/TabPFNGamma_${size}_${trial}.log" 2>&1 \
                    || echo "[GPU${gpu_id}] ERROR ${size}/${trial}/Gamma  — see log"
                echo "[GPU${gpu_id}] DONE  ${size}/${trial}/Gamma  — $(date +"%T")"
            fi
        done
    done
}

# ── Dispatch: 5 trials per GPU ────────────────────────────────────────────────
run_gpu_worker 0 trial_00 trial_01 trial_02 trial_03 trial_04 &
run_gpu_worker 1 trial_05 trial_06 trial_07 trial_08 trial_09 &
run_gpu_worker 2 trial_10 trial_11 trial_12 trial_13 trial_14 &
run_gpu_worker 3 trial_15 trial_16 trial_17 trial_18 trial_19 &

wait

echo ""
echo "All workers finished at: $(date +"%T")"
echo "Run check_and_rerun_failed.sh to verify and fix any failures."
