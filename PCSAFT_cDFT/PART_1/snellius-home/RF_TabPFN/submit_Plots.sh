#!/bin/bash

#SBATCH -J Plot_RF_TabPFN
#SBATCH -t 01:00:00
#SBATCH -p thin
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH -o /gpfs/home6/draju/A6/TabPFN/RF_TabPFN/slurm-plots-%j.out

set -euo pipefail

echo "Plot job started at: $(date +"%T")"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load 2025
module load Python/3.13.1-GCCcore-14.2.0

cd /gpfs/home6/draju/A6/TabPFN/RF_TabPFN
source /gpfs/home6/draju/A6/.A6/bin/activate

export MPLBACKEND=Agg

OUTPUT_ROOT="/gpfs/home6/draju/A6/TabPFN/RF_TabPFN"

run_plot() {
    local notebook="$1"
    local output_name="$2"
    local plot_folder="$3"

    local output_dir="${OUTPUT_ROOT}/${plot_folder}"
    mkdir -p "${output_dir}"

    echo "Starting ${notebook} at $(date +"%T")"
    papermill "${notebook}" "${output_dir}/${output_name}_output.ipynb" \
        -p PLOT_FOLDER "${output_dir}"
    echo "Finished ${notebook} at $(date +"%T")"
}

run_plot RF_TabPFNBubble_Plots.ipynb RF_TabPFNBubble_Plots SLURMBubble &
pid_bubble=$!

run_plot RF_TabPFNDew_Plots.ipynb RF_TabPFNDew_Plots SLURMDew &
pid_dew=$!

run_plot RF_TabPFNGamma_Plots.ipynb RF_TabPFNGamma_Plots SLURMGamma &
pid_gamma=$!

run_plot RF_TabPFNThickness_Plots.ipynb RF_TabPFNThickness_Plots SLURMTHICKNESS &
pid_thickness=$!

wait "${pid_bubble}" "${pid_dew}" "${pid_gamma}" "${pid_thickness}"

echo "Plot job finished at: $(date +"%T")"
