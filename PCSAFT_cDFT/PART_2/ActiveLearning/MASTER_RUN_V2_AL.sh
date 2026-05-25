#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# MASTER_RUN_V2_AL.sh
#
# Submits SLURM array jobs for the V2_AL active-learning runs only.
# Loops over 4 sizes × 20 trials = 80 parent jobs; each parent expands into
# an array of N composition tasks via RUN_trial.sh.
#
# Total expanded tasks: 20 × (25 + 50 + 75 + 100) = 5000.
#
# Usage:
#   ./MASTER_RUN_V2_AL.sh           # submit all 80 parent jobs
#   ./MASTER_RUN_V2_AL.sh --dry     # preview the sbatch commands only
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIAL_SCRIPT="${SCRIPT_DIR}/RUN_trial.sh"

METHOD="AL_V2_AL"
SIZES=(N025 N050 N075 N100)
TRIALS=(trial_00 trial_01 trial_02 trial_03 trial_04
        trial_05 trial_06 trial_07 trial_08 trial_09
        trial_10 trial_11 trial_12 trial_13 trial_14
        trial_15 trial_16 trial_17 trial_18 trial_19)

DRY_RUN=0
if [[ "${1:-}" == "--dry" ]] || [[ "${1:-}" == "-n" ]]; then
    DRY_RUN=1
fi

TOTAL=$((${#SIZES[@]} * ${#TRIALS[@]}))

echo "Master run (${METHOD}) started at: $(date '+%F %T')"
echo "Sizes  : ${SIZES[*]}"
echo "Trials : ${#TRIALS[@]}"
echo "Total parent submissions: ${TOTAL}"
if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "MODE   : DRY-RUN (no sbatch calls will be made)"
fi
echo ""

if [[ ! -x "${TRIAL_SCRIPT}" ]]; then
    echo "Error: ${TRIAL_SCRIPT} not found or not executable" >&2
    exit 1
fi

N_SUBMITTED=0
N_SKIPPED=0

for SIZE in "${SIZES[@]}"; do
    for TRIAL in "${TRIALS[@]}"; do
        CSV="${SCRIPT_DIR}/${METHOD}/OUTPUT/${SIZE}/${TRIAL}.csv"

        if [[ ! -f "${CSV}" ]]; then
            echo "Warning: ${CSV#${SCRIPT_DIR}/} not found — skipping"
            N_SKIPPED=$((N_SKIPPED + 1))
            continue
        fi

        if [[ ${DRY_RUN} -eq 1 ]]; then
            echo "  [dry] sbatch RUN_trial.sh ${METHOD} ${SIZE} ${TRIAL}"
        else
            echo "Submitting: ${METHOD} / ${SIZE} / ${TRIAL}"
            sbatch --chdir="${SCRIPT_DIR}" "${TRIAL_SCRIPT}" "${METHOD}" "${SIZE}" "${TRIAL}"
        fi
        N_SUBMITTED=$((N_SUBMITTED + 1))
    done
done

echo ""
echo "Master run completed at: $(date '+%F %T')"
if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "Would submit: ${N_SUBMITTED}    Skipped (missing CSV): ${N_SKIPPED}"
else
    echo "Submitted: ${N_SUBMITTED}    Skipped (missing CSV): ${N_SKIPPED}"
    echo ""
    echo "Monitor with:  squeue -u \$USER"
fi
