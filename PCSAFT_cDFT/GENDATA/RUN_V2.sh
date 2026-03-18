#!/bin/bash
set -e

#SBATCH --job-name=cDFT_TERNARY-(CO2-H2-Ar)
#SBATCH --partition=serial
#SBATCH --time=6:00:00
#SBATCH --exclude=c171
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

start_time=$(date +"%T")
echo "Job started at: $start_time"

source  /home/darshan/A6/py_A6/bin/activate
export PATH=$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH

which python
which latex

mkdir -p ${SLURM_JOB_ID}

# VLE_IFT_V2.ipynb
papermill VLE_IFT_V2.ipynb ${SLURM_JOB_ID}/VLE_IFT_V2_${SLURM_JOB_ID}.ipynb \
-p COMPONENTS '["carbon dioxide","hydrogen","argon"]' \
-p KIJ_map '{"CO2,Ar":"zero","H2,Ar":"zero","CO2,H2":"zero"}' \
-p CO2_comp 0.99 \
-p n_feeds 2 \
-p verbose False \
-p CSV_FOLDER "CSV_${SLURM_JOB_ID}" \
-p PLOT_FOLDER "PLOTS_${SLURM_JOB_ID}" \
-p SLURM_RUN True \
