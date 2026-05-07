#!/bin/bash
set -euo pipefail

# Auto-generated resubmit script for failed tasks.
# Each line submits only the specific task IDs that failed.

echo "Resubmitting failed tasks across trials..."

sbatch --array=22,23     RUN_trial.sh N025 trial_11
sbatch --array=19        RUN_trial.sh N025 trial_17
sbatch --array=13        RUN_trial.sh N025 trial_19
sbatch --array=30        RUN_trial.sh N050 trial_01
sbatch --array=12        RUN_trial.sh N050 trial_06
sbatch --array=7         RUN_trial.sh N075 trial_05
sbatch --array=17,22     RUN_trial.sh N075 trial_08
sbatch --array=19        RUN_trial.sh N075 trial_16
sbatch --array=11,22     RUN_trial.sh N075 trial_18
sbatch --array=11,12     RUN_trial.sh N100 trial_01
sbatch --array=52        RUN_trial.sh N100 trial_10
sbatch --array=68,69     RUN_trial.sh N100 trial_13
sbatch --array=0         RUN_trial.sh N100 trial_14
sbatch --array=4,22,23,24,88,89,90,91,99  RUN_trial.sh N100 trial_16

echo "Done. Run clean.ipynb again after jobs complete to verify."
