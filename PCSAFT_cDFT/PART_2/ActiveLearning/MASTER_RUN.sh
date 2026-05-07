#!/bin/bash
set -euo pipefail

# Submit VLE+IFT array jobs for all AL types and sizes.
# Usage: bash MASTER_RUN.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AL_SCRIPT="${SCRIPT_DIR}/RUN_AL.sh"

AL_TYPES=(AL_ST AL_MT)
SIZES=(N025 N050 N075 N100)

submitted=0
skipped=0

for AL_TYPE in "${AL_TYPES[@]}"; do
    for SIZE in "${SIZES[@]}"; do
        CSV="${SCRIPT_DIR}/${AL_TYPE}/AL/AL_${SIZE}.csv"
        if [ ! -f "${CSV}" ]; then
            echo "SKIP: ${AL_TYPE}/AL/AL_${SIZE}.csv not found"
            skipped=$((skipped + 1))
            continue
        fi
        echo "Submitting ${AL_TYPE} ${SIZE}..."
        sbatch --chdir="${SCRIPT_DIR}" "${AL_SCRIPT}" "${AL_TYPE}" "${SIZE}"
        submitted=$((submitted + 1))
    done
done

echo ""
echo "Submitted : ${submitted} array jobs  (2 AL types x 4 sizes)"
echo "Skipped   : ${skipped} (CSV not found)"
echo ""
echo "Run clean.ipynb after jobs complete to combine results."
