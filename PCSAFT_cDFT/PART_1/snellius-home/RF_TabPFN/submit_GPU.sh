#!/bin/bash

#SBATCH -J GPU_RF_TabPFN
#SBATCH -t 24:00:00
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=720G
#SBATCH --gres=gpu:4
#SBATCH -o /gpfs/home6/draju/A6/TabPFN/RF_TabPFN/slurm-calc-%j.out

set -euo pipefail

start_time=$(date +"%T")
echo "Job started at: ${start_time}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load 2025
module load MPICH/4.3.0-GCC-14.2.0-CUDA-12.8.0
module load Python/3.13.1-GCCcore-14.2.0

cd /gpfs/home6/draju/A6/TabPFN/RF_TabPFN
source /gpfs/home6/draju/A6/.A6/bin/activate

echo "CUDA Version: ${EBVERSIONCUDA}"
echo "MPICH Version: ${EBVERSIONMPICH}"
echo "Python Version: ${EBVERSIONPYTHON}"
echo "LaTeX : $(which latex || echo 'latex not found')"

OUTPUT_ROOT="/gpfs/home6/draju/A6/TabPFN/RF_TabPFN"

TEST_ROWS="${TEST_ROWS:-None}"
echo "TEST_ROWS=${TEST_ROWS}"

run_notebook() {
    local gpu_id="$1"
    local notebook="$2"
    local output_name="$3"
    local plot_folder="$4"
    local seed="$5"

    local output_dir="${OUTPUT_ROOT}/${plot_folder}"
    mkdir -p "${output_dir}"

    echo "Starting ${notebook} on GPU ${gpu_id} at $(date +"%T")"
    CUDA_VISIBLE_DEVICES="${gpu_id}" papermill "${notebook}" "${output_dir}/${output_name}_output.ipynb" \
        -p PLOT_FOLDER "${output_dir}" \
        -p TEST_ROWS "${TEST_ROWS}" \
        -p SEED "${seed}"
    echo "Finished ${notebook} on GPU ${gpu_id} at $(date +"%T")"
}

run_notebook 0 RF_TabPFNBubble_Calc.ipynb RF_TabPFNBubble_Calc SLURMBubble 50015 &
pid_bubble=$!

run_notebook 1 RF_TabPFNDew_Calc.ipynb RF_TabPFNDew_Calc SLURMDew 855015 &
pid_dew=$!

run_notebook 2 RF_TabPFNGamma_Calc.ipynb RF_TabPFNGamma_Calc SLURMGamma 50005 &
pid_gamma=$!

run_notebook 3 RF_TabPFNThickness_Calc.ipynb RF_TabPFNThickness_Calc SLURMTHICKNESS 655552 &
pid_thickness=$!

wait "${pid_bubble}" "${pid_dew}" "${pid_gamma}" "${pid_thickness}"

echo "Job finished at: $(date +"%T")"
