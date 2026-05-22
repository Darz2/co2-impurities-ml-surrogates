#!/bin/bash

#SBATCH -J GPU_TabPFN_Verify
#SBATCH -t 02:00:00
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=720G
#SBATCH --gres=gpu:4
#SBATCH -o /gpfs/home6/draju/A6/TabPFN_V3/with_HPO/slurm-verify-%j.out

set -euo pipefail

start_time=$(date +"%T")
echo "Job started at: ${start_time}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load 2025
module load MPICH/4.3.0-GCC-14.2.0-CUDA-12.8.0
module load Python/3.13.1-GCCcore-14.2.0

cd /gpfs/home6/draju/A6/TabPFN_V3/with_HPO
source /gpfs/home6/draju/A6/.TPFN/bin/activate

echo "CUDA Version: ${EBVERSIONCUDA}"
echo "Python Version: ${EBVERSIONPYTHON}"

OUTPUT_ROOT="/gpfs/home6/draju/A6/TabPFN_V3/with_HPO"

TEST_ROWS="${TEST_ROWS:-None}"

run_notebook() {
    local gpu_id="$1"
    local notebook="$2"
    local output_name="$3"
    local plot_folder="$4"
    local hpo_folder="$5"
    local seed="$6"

    local output_dir="${OUTPUT_ROOT}/${plot_folder}"
    mkdir -p "${output_dir}"

    echo "Starting ${notebook} on GPU ${gpu_id} at $(date +"%T")"
    CUDA_VISIBLE_DEVICES="${gpu_id}" papermill "${notebook}" "${output_dir}/${output_name}_output.ipynb" \
        -p PLOT_FOLDER "${output_dir}" \
        -p HPO_FOLDER  "${OUTPUT_ROOT}/${hpo_folder}" \
        -p TEST_ROWS   "${TEST_ROWS}" \
        -p SEED        "${seed}"
    echo "Finished ${notebook} on GPU ${gpu_id} at $(date +"%T")"
}

run_notebook 0 TabPFNBubble_HPO_Verify.ipynb    TabPFNBubble_HPO_Verify    SLURMBubble_Verify    SLURMBubble    454015  &
pid_bubble=$!

run_notebook 1 TabPFNDew_HPO_Verify.ipynb       TabPFNDew_HPO_Verify       SLURMDew_Verify       SLURMDew       6702315 &
pid_dew=$!

run_notebook 2 TabPFNGamma_HPO_Verify.ipynb     TabPFNGamma_HPO_Verify     SLURMGamma_Verify     SLURMGamma     844015  &
pid_gamma=$!

run_notebook 3 TabPFNThickness_HPO_Verify.ipynb TabPFNThickness_HPO_Verify SLURMTHICKNESS_Verify SLURMTHICKNESS 655552  &
pid_thickness=$!

wait "${pid_bubble}" "${pid_dew}" "${pid_gamma}" "${pid_thickness}"

echo "Job finished at: $(date +"%T")"
