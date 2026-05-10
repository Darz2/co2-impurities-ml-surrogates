#!/bin/bash

#SBATCH -J XGB_RERUN_AL
#SBATCH -t 00:30:00
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH -o /scratch-shared/draju/PART_2/ACTIVELEARNING/slurm-xgb-rerun-%j.out

# Reruns 6 missing XGB notebooks for ACTIVELEARNING/AL_ST (kernel-startup timeouts).
#   AL_ST/N025/trial_00/XGBBubble
#   AL_ST/N050/trial_01/XGBGamma
#   AL_ST/N050/trial_09/XGBBubble
#   AL_ST/N050/trial_15/XGBGamma
#   AL_ST/N075/trial_04/XGBBubble
#   AL_ST/N100/trial_02/XGBGamma

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
THREADS_PER_JOB=4

run_bubble() {
    local al_type="$1" trial="$2" size="$3"
    local trial_idx=$(( 10#${trial##trial_} ))
    local seed=$(( SEED_BUBBLE + trial_idx ))
    local out_dir="${OUTPUT_ROOT}/${al_type}/${size}/${trial}"
    mkdir -p "${out_dir}/XGBBubble"
    local nb_out="${out_dir}/XGBBubble_${al_type}_${size}_${trial}_output.ipynb"
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

run_bubble AL_ST trial_00 N025 &
run_gamma  AL_ST trial_01 N050 &
run_bubble AL_ST trial_09 N050 &
run_gamma  AL_ST trial_15 N050 &
run_bubble AL_ST trial_04 N075 &
run_gamma  AL_ST trial_02 N100 &
wait

echo ""
echo "Rerun finished at: $(date +"%T")"
