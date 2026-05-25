#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# RERUN_V2_AL.sh
#
# Resubmits the V2_AL tasks that failed in the first MASTER_RUN_V2_AL.sh pass,
# as identified by clean_trials.ipynb's "=== Ran but failed ===" report.
#
# Run from within ActiveLearning/:  ./RERUN_V2_AL.sh
# Use --dry / -n to preview without submitting.
#
# Each entry below corresponds to one (size, trial, task_id) tuple from the
# clean_trials report. Add or remove lines as failures evolve across runs.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIAL_SCRIPT="${SCRIPT_DIR}/RUN_trial.sh"
METHOD="AL_V2_AL"

DRY_RUN=0
if [[ "${1:-}" == "--dry" ]] || [[ "${1:-}" == "-n" ]]; then
    DRY_RUN=1
fi

# (SIZE, TRIAL, TASK_ID) tuples that failed and need resubmission
declare -a JOBS=(
    "N025 trial_01 8"
    "N050 trial_02 35"
    "N050 trial_03 44"
    "N100 trial_14 1"
)

echo "RERUN_V2_AL started at: $(date '+%F %T')"
echo "Resubmitting ${#JOBS[@]} failed ${METHOD} task(s)"
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

for ENTRY in "${JOBS[@]}"; do
    read -r SIZE TRIAL TASK_ID <<< "${ENTRY}"
    CSV="${SCRIPT_DIR}/${METHOD}/OUTPUT/${SIZE}/${TRIAL}.csv"

    if [[ ! -f "${CSV}" ]]; then
        echo "Warning: ${CSV#${SCRIPT_DIR}/} not found — skipping ${SIZE}/${TRIAL}/[${TASK_ID}]"
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi

    if [[ ${DRY_RUN} -eq 1 ]]; then
        echo "  [dry] sbatch --array=${TASK_ID} RUN_trial.sh ${METHOD} ${SIZE} ${TRIAL}"
    else
        echo "Resubmitting: ${METHOD} / ${SIZE} / ${TRIAL}  task=${TASK_ID}"
        sbatch --array="${TASK_ID}" --chdir="${SCRIPT_DIR}" \
               "${TRIAL_SCRIPT}" "${METHOD}" "${SIZE}" "${TRIAL}"
    fi
    N_SUBMITTED=$((N_SUBMITTED + 1))
done

echo ""
echo "RERUN_V2_AL completed at: $(date '+%F %T')"
if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "Would submit: ${N_SUBMITTED}    Skipped (missing CSV): ${N_SKIPPED}"
else
    echo "Submitted: ${N_SUBMITTED}    Skipped (missing CSV): ${N_SKIPPED}"
    echo ""
    echo "After completion, rerun clean_trials.ipynb to verify."
fi
