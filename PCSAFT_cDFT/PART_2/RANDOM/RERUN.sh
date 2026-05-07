#!/bin/bash
set -euo pipefail

# Auto-generated resubmit script for failed tasks.
# Each line submits only the specific task IDs that failed.

echo "Resubmitting failed tasks across trials..."

sbatch --array=14        RUN_trial.sh N025 trial_14
sbatch --array=47        RUN_trial.sh N050 trial_00
sbatch --array=8         RUN_trial.sh N050 trial_06
sbatch --array=41        RUN_trial.sh N050 trial_08
sbatch --array=43,49     RUN_trial.sh N050 trial_15
sbatch --array=4,5       RUN_trial.sh N050 trial_16
sbatch --array=14,20     RUN_trial.sh N075 trial_01
sbatch --array=55        RUN_trial.sh N075 trial_02
sbatch --array=58,59,60  RUN_trial.sh N075 trial_03
sbatch --array=35        RUN_trial.sh N075 trial_04
sbatch --array=40,41     RUN_trial.sh N075 trial_10
sbatch --array=58,60     RUN_trial.sh N075 trial_11
sbatch --array=23        RUN_trial.sh N075 trial_14
sbatch --array=10        RUN_trial.sh N075 trial_17
sbatch --array=81        RUN_trial.sh N100 trial_02
sbatch --array=13,14,15,16,17,37   RUN_trial.sh N100 trial_03
sbatch --array=39        RUN_trial.sh N100 trial_05
sbatch --array=34        RUN_trial.sh N100 trial_07
sbatch --array=10,11     RUN_trial.sh N100 trial_10
sbatch --array=89        RUN_trial.sh N100 trial_11
sbatch --array=53,54,55,85,86,87   RUN_trial.sh N100 trial_13
sbatch --array=22,63,64  RUN_trial.sh N100 trial_15

echo "Done. Run clean.ipynb again after jobs complete to verify."
