#!/bin/bash

#SBATCH -J XGB_RERUN_RANDOM
#SBATCH -t 00:30:00
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH -o /scratch-shared/draju/PART_2/RANDOM/slurm-xgb-rerun-%j.out

# Reruns 2 missing XGB notebooks for RANDOM (kernel-startup timeouts).
#   N075/trial_09/XGBGamma
#   N100/trial_00/XGBGamma

set -uo pipefail

echo "Job started at: $(date +"%T")"

module load 2025
module load Python/3.13.1-GCCcore-14.2.0

cd /scratch-shared/draju/PART_2/RANDOM
source /gpfs/home6/draju/A6/.A6/bin/activate

DATA_ROOT="/scratch-shared/draju/PART_2/RANDOM/COMBINED"
OUTPUT_ROOT="/scratch-shared/draju/PART_2/RANDOM/XGB_OUTPUTS"
SEED_BUBBLE=50015
SEED_GAMMA=50005
THREADS_PER_JOB=4

run_bubble() {
    local trial="$1" size="$2"
    local trial_idx=$(( 10#${trial##trial_} ))
    local seed=$(( SEED_BUBBLE + trial_idx ))
    local out_dir="${OUTPUT_ROOT}/${size}/${trial}"
    mkdir -p "${out_dir}/XGBBubble"
    local nb_out="${out_dir}/XGBBubble_${size}_${trial}_output.ipynb"
    echo "START ${size}/${trial}/XGBBubble (seed=${seed}) — $(date +"%T")"
    SLURM_CPUS_PER_TASK=${THREADS_PER_JOB} papermill --start-timeout 120 XGBBubble.ipynb \
        "$nb_out" \
        -p OUTPUT_FOLDER "${out_dir}/XGBBubble" \
        -p TEST_ROWS     None \
        -p SEED          "${seed}" \
        -p DATA_PATH     "${DATA_ROOT}/${size}/${trial}.csv" \
        > "${out_dir}/XGBBubble_${size}_${trial}.log" 2>&1 \
        && echo "DONE  ${size}/${trial}/XGBBubble — $(date +"%T")" \
        || echo "ERROR ${size}/${trial}/XGBBubble — see log"
}

run_gamma() {
    local trial="$1" size="$2"
    local trial_idx=$(( 10#${trial##trial_} ))
    local seed=$(( SEED_GAMMA + trial_idx ))
    local out_dir="${OUTPUT_ROOT}/${size}/${trial}"
    mkdir -p "${out_dir}/XGBGamma"
    local nb_out="${out_dir}/XGBGamma_${size}_${trial}_output.ipynb"
    echo "START ${size}/${trial}/XGBGamma  (seed=${seed}) — $(date +"%T")"
    SLURM_CPUS_PER_TASK=${THREADS_PER_JOB} papermill --start-timeout 120 XGBGamma.ipynb \
        "$nb_out" \
        -p OUTPUT_FOLDER "${out_dir}/XGBGamma" \
        -p TEST_ROWS     None \
        -p SEED          "${seed}" \
        -p DATA_PATH     "${DATA_ROOT}/${size}/${trial}.csv" \
        > "${out_dir}/XGBGamma_${size}_${trial}.log" 2>&1 \
        && echo "DONE  ${size}/${trial}/XGBGamma  — $(date +"%T")" \
        || echo "ERROR ${size}/${trial}/XGBGamma  — see log"
}

run_gamma trial_09 N075 &
run_gamma trial_00 N100 &
wait

echo ""
echo "Rerun finished at: $(date +"%T")"
