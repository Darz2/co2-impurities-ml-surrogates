#!/bin/bash
# Resubmit failed ActiveLearning tasks identified by clean.ipynb.
# Run from within ActiveLearning/: bash RERUN.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AL_SCRIPT="${SCRIPT_DIR}/RUN_AL.sh"

echo "Resubmitting failed AL tasks..."

sbatch --array=12,13,14,15,16 --chdir="${SCRIPT_DIR}" "${AL_SCRIPT}" AL_MT N050
sbatch --array=2,67,85        --chdir="${SCRIPT_DIR}" "${AL_SCRIPT}" AL_MT N100
sbatch --array=49,50,51,52    --chdir="${SCRIPT_DIR}" "${AL_SCRIPT}" AL_ST N075

echo ""
echo "Submitted 3 rerun array jobs."
echo "Run clean.ipynb again after jobs complete to verify."
