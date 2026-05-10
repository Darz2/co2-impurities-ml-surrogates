#!/bin/bash

#SBATCH -J Plot_TabPFN
#SBATCH -t 01:00:00
#SBATCH -p thin
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH -o /gpfs/home6/draju/A6/TabPFN/without_HPO/slurm-plots-%j.out

set -euo pipefail

echo "Plot job started at: $(date +"%T")"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load 2025
module load Python/3.13.1-GCCcore-14.2.0

cd /gpfs/home6/draju/A6/TabPFN/without_HPO
source /gpfs/home6/draju/A6/.A6/bin/activate

export MPLBACKEND=Agg

OUTPUT_ROOT="/gpfs/home6/draju/A6/TabPFN/without_HPO"

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

run_plot TabPFNBubble_Plots.ipynb TabPFNBubble_Plots SLURMBubble &
pid_bubble=$!

run_plot TabPFNDew_Plots.ipynb TabPFNDew_Plots SLURMDew &
pid_dew=$!

run_plot TabPFNGamma_Plots.ipynb TabPFNGamma_Plots SLURMGamma &
pid_gamma=$!

run_plot TabPFNThickness_Plots.ipynb TabPFNThickness_Plots SLURMTHICKNESS &
pid_thickness=$!

wait "${pid_bubble}" "${pid_dew}" "${pid_gamma}" "${pid_thickness}"

echo "Plot job finished at: $(date +"%T")"
