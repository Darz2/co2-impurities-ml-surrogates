#!/bin/bash

#SBATCH -J AL_XGB_ALL
#SBATCH -t 08:00:00
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH -o /scratch-shared/draju/PART_2/ACTIVELEARNING/slurm-xgb-all-trials-%j.out

# Runs all 2 AL types × 20 trials × 4 sizes × XGBBubble + XGBGamma (640 notebooks).
# Uses the full 192-CPU genoa node: 48 parallel papermill processes × 4 threads each.
# Already-completed notebooks (output file > 50 KB) are skipped.

set -uo pipefail

echo "Job started at: $(date +"%T")"

module load 2025
module load Python/3.13.1-GCCcore-14.2.0

cd /scratch-shared/draju/PART_2/ACTIVELEARNING
source /gpfs/home6/draju/A6/.A6/bin/activate

DATA_ROOT="/scratch-shared/draju/PART_2/ACTIVELEARNING/COMBINED"
OUTPUT_ROOT="/scratch-shared/draju/PART_2/ACTIVELEARNING/XGB_OUTPUTS"
SEED_BUBBLE=50015
SEED_GAMMA=52225

AL_TYPES=(AL_ST AL_MT)
SIZES=(N025 N050 N075 N100)
TRIALS=()
for i in $(seq 0 19); do TRIALS+=( "$(printf 'trial_%02d' "$i")" ); done

THREADS_PER_JOB=4
N_PARALLEL=$(( SLURM_CPUS_PER_TASK / THREADS_PER_JOB ))
echo "Pool: ${N_PARALLEL} parallel jobs × ${THREADS_PER_JOB} threads = ${SLURM_CPUS_PER_TASK} CPUs"

# ── Pool: reap finished PIDs; block until a slot is free ──────────────────────
PIDS=()
pool_wait() {
    while (( ${#PIDS[@]} >= N_PARALLEL )); do
        local live=()
        for p in "${PIDS[@]}"; do
            if kill -0 "$p" 2>/dev/null; then live+=("$p"); else wait "$p" || true; fi
        done
        PIDS=("${live[@]+"${live[@]}"}")
        (( ${#PIDS[@]} >= N_PARALLEL )) && sleep 1
    done
}

# ── Single-notebook runner (called in background) ─────────────────────────────
# Seed varies per trial: BASE_SEED + trial_index (trial_07 → +7, etc.)
run_bubble() {
    local al_type="$1" trial="$2" size="$3"
    local trial_idx=$(( 10#${trial##trial_} ))
    local seed=$(( SEED_BUBBLE + trial_idx ))
    local out_dir="${OUTPUT_ROOT}/${al_type}/${size}/${trial}"
    mkdir -p "${out_dir}/XGBBubble"
    local nb_out="${out_dir}/XGBBubble_${al_type}_${size}_${trial}_output.ipynb"
    if [[ -f "$nb_out" ]] && (( $(stat -c%s "$nb_out") > 50000 )); then
        echo "SKIP  ${al_type}/${size}/${trial}/XGBBubble"
        return 0
    fi
    echo "START ${al_type}/${size}/${trial}/XGBBubble (seed=${seed}) — $(date +"%T")"
    SLURM_CPUS_PER_TASK=${THREADS_PER_JOB} papermill --start-timeout 120 XGBBubble.ipynb \
        "$nb_out" \
        -p OUTPUT_FOLDER "${out_dir}/XGBBubble" \
        -p TEST_ROWS     None \
        -p SEED          "${seed}" \
        -p DATA_PATH     "${DATA_ROOT}/${al_type}/${size}/${trial}.csv" \
        > "${out_dir}/XGBBubble_${al_type}_${size}_${trial}.log" 2>&1 \
        && echo "DONE  ${al_type}/${size}/${trial}/XGBBubble — $(date +"%T")" \
        || echo "ERROR ${al_type}/${size}/${trial}/XGBBubble — see log"
}

run_gamma() {
    local al_type="$1" trial="$2" size="$3"
    local trial_idx=$(( 10#${trial##trial_} ))
    local seed=$(( SEED_GAMMA + trial_idx ))
    local out_dir="${OUTPUT_ROOT}/${al_type}/${size}/${trial}"
    mkdir -p "${out_dir}/XGBGamma"
    local nb_out="${out_dir}/XGBGamma_${al_type}_${size}_${trial}_output.ipynb"
    if [[ -f "$nb_out" ]] && (( $(stat -c%s "$nb_out") > 50000 )); then
        echo "SKIP  ${al_type}/${size}/${trial}/XGBGamma"
        return 0
    fi
    echo "START ${al_type}/${size}/${trial}/XGBGamma  (seed=${seed}) — $(date +"%T")"
    SLURM_CPUS_PER_TASK=${THREADS_PER_JOB} papermill --start-timeout 120 XGBGamma.ipynb \
        "$nb_out" \
        -p OUTPUT_FOLDER "${out_dir}/XGBGamma" \
        -p TEST_ROWS     None \
        -p SEED          "${seed}" \
        -p DATA_PATH     "${DATA_ROOT}/${al_type}/${size}/${trial}.csv" \
        > "${out_dir}/XGBGamma_${al_type}_${size}_${trial}.log" 2>&1 \
        && echo "DONE  ${al_type}/${size}/${trial}/XGBGamma  — $(date +"%T")" \
        || echo "ERROR ${al_type}/${size}/${trial}/XGBGamma  — see log"
}

# ── Dispatch all 640 jobs into the pool ───────────────────────────────────────
for al_type in "${AL_TYPES[@]}"; do
    for trial in "${TRIALS[@]}"; do
        for size in "${SIZES[@]}"; do
            pool_wait; run_bubble "$al_type" "$trial" "$size" & PIDS+=($!)
            pool_wait; run_gamma  "$al_type" "$trial" "$size" & PIDS+=($!)
        done
    done
done

wait
echo ""
echo "All jobs finished at: $(date +"%T")"
