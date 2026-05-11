#!/bin/bash

#SBATCH -J Plot_TabPFN
#SBATCH -t 00:30:00
#SBATCH -p thin
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH -o /gpfs/home6/draju/A6/TabPFN/SLURM_Pbubble/slurm-%j.out

set -euo pipefail

echo "Plot job started at: $(date +"%T")"

cd /gpfs/home6/draju/A6/TabPFN
source /gpfs/home6/draju/A6/.A6/bin/activate

module load 2025
module load Python/3.13.1-GCCcore-14.2.0

OUTPUT_DIR="/gpfs/home6/draju/A6/TabPFN/SLURM_Pbubble"

papermill TabPFNBubble_Plot.ipynb "${OUTPUT_DIR}/TabPFNBubble_Plot_output.ipynb" \
    -p PLOT_FOLDER "${OUTPUT_DIR}" \
    -p SEED 454015

echo "Plot job finished at: $(date +"%T")"
