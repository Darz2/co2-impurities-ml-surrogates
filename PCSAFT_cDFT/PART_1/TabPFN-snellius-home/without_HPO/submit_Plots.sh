#!/bin/bash

#SBATCH --job-name=TabPFN-PLOTS
#SBATCH --partition=highmem
#SBATCH --time=12:00:00
#SBATCH --nodelist=c108
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

set -euo pipefail

echo "Plot job started at: $(date +"%T")"

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

source /home/darshan/A6/py_A6/bin/activate
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"
export MPLBACKEND=Agg

echo "Python: $(which python)"
echo "LaTeX : $(which latex || echo 'latex not found')"

# Maps: notebook  output_base  plot_folder (where inputs live and outputs go)
tasks=(
    "TabPFNBubble_Plots.ipynb     TabPFNBubble_Plots     TabPFN_P_bubble_OUTPUTS"
    "TabPFNDew_Plots.ipynb        TabPFNDew_Plots        TabPFN_P_dew_OUTPUTS"
    "TabPFNGamma_Plots.ipynb      TabPFNGamma_Plots      TabPFN_gamma_OUTPUTS"
    "TabPFNThickness_Plots.ipynb  TabPFNThickness_Plots  TabPFN_interfacial_thickness_OUTPUTS"
)

run_plot() {
    local notebook="$1"
    local output_name="$2"
    local plot_folder="$3"
    local log_file="${plot_folder}/${output_name}.log"

    mkdir -p "${plot_folder}"

    echo "[$(date +"%T")] Starting ${notebook} (logs -> ${log_file})"
    papermill "${notebook}" "${plot_folder}/${output_name}_output.ipynb" \
        -p PLOT_FOLDER "${plot_folder}" \
        > "${log_file}" 2>&1
    echo "[$(date +"%T")] Finished ${notebook}"
}

declare -a pids names
for task in "${tasks[@]}"; do
    read -r notebook output_name plot_folder <<< "${task}"
    run_plot "${notebook}" "${output_name}" "${plot_folder}" &
    pids+=("$!")
    names+=("${notebook}")
done

# Wait for each background job individually so no failures get swallowed
exit_code=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[$(date +"%T")] OK    ${names[$i]}"
    else
        rc=$?
        echo "[$(date +"%T")] FAIL  ${names[$i]} (exit ${rc})"
        exit_code=1
    fi
done

echo "Plot job finished at: $(date +"%T") (exit ${exit_code})"
exit "${exit_code}"
