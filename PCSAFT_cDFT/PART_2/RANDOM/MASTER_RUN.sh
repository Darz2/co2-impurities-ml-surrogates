#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIAL_SCRIPT="${SCRIPT_DIR}/RUN_trial.sh"

SIZES=(N025 N050 N075 N100)
TRIALS=(trial_00 trial_01 trial_02 trial_03 trial_04
        trial_05 trial_06 trial_07 trial_08 trial_09
        trial_10 trial_11 trial_12 trial_13 trial_14
        trial_15 trial_16 trial_17 trial_18 trial_19)

echo "Master run started at: $(date '+%F %T')"
echo "Sizes: ${SIZES[*]}"
echo "Trials per size: ${#TRIALS[@]}"
echo "Total array submissions: $((${#SIZES[@]} * ${#TRIALS[@]}))"
echo ""

N_SUBMITTED=0
N_SKIPPED=0

for SIZE in "${SIZES[@]}"; do
    for TRIAL in "${TRIALS[@]}"; do
        CSV="${SCRIPT_DIR}/OUTPUT/${SIZE}/${TRIAL}.csv"
        if [ ! -f "${CSV}" ]; then
            echo "Warning: ${CSV} not found — skipping"
            N_SKIPPED=$((N_SKIPPED + 1))
            continue
        fi
        echo "Submitting: ${SIZE} / ${TRIAL}"
        sbatch --chdir="${SCRIPT_DIR}" "${TRIAL_SCRIPT}" "${SIZE}" "${TRIAL}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    done
done

echo ""
echo "Master run completed at: $(date '+%F %T')"
echo "Submitted: ${N_SUBMITTED}  Skipped: ${N_SKIPPED}"
