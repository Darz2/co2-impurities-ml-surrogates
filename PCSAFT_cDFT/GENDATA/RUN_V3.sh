#!/bin/bash
set -e

#SBATCH --job-name=cDFT_V3
#SBATCH --partition=serial
#SBATCH --time=6:00:00
#SBATCH --exclude=c171
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-99  # Adjust range based on number of compositions in CSV

start_time=$(date +"%T")
echo "Job started at: $start_time"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

source /home/darshan/A6/py_A6/bin/activate
export PATH=$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH

which python
which latex

# Create output directory for this array task
OUTPUT_DIR="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p ${OUTPUT_DIR}

# VLE_IFT_V3.ipynb - Run single composition based on array task ID
papermill VLE_IFT_V3.ipynb ${OUTPUT_DIR}/VLE_IFT_V3_${SLURM_ARRAY_TASK_ID}.ipynb \
    -p FEED_INDEX ${SLURM_ARRAY_TASK_ID} \
    -p SLURM_RUN True \
    -p verbose False \
    -p CSV_FOLDER "${OUTPUT_DIR}/CSV" \
    -p PLOT_FOLDER "${OUTPUT_DIR}/PLOTS"

echo "Job finished at: $(date +"%T")"
