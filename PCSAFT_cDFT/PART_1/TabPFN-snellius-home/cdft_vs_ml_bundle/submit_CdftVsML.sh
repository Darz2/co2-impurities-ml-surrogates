#!/bin/bash

#SBATCH -J CdftVsML_Timing
#SBATCH -t 02:00:00
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /gpfs/home6/draju/A6/TabPFN/cdft_vs_ml_bundle/slurm-cdft-%j.out

set -euo pipefail

start_time=$(date +"%T")
echo "Job started at: ${start_time}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load 2025
module load MPICH/4.3.0-GCC-14.2.0-CUDA-12.8.0
module load Python/3.13.1-GCCcore-14.2.0

cd /gpfs/home6/draju/A6/TabPFN/cdft_vs_ml_bundle
source /gpfs/home6/draju/A6/.A6/bin/activate

# The notebook's embedded kernelspec is named "py_a6"; register that name from THIS
# (.A6) env so papermill can find it on this HPC (fixes: NoSuchKernel py_a6).
python -m ipykernel install --user --name py_a6 --display-name py_a6

echo "CUDA Version: ${EBVERSIONCUDA}"
echo "MPICH Version: ${EBVERSIONMPICH}"
echo "Python Version: ${EBVERSIONPYTHON}"

echo "Running cdft_vs_ml_timing.ipynb on GPU 0 at $(date +"%T")"
CUDA_VISIBLE_DEVICES=0 papermill cdft_vs_ml_timing.ipynb cdft_vs_ml_timing_output.ipynb
echo "Finished at $(date +"%T")"

echo "Job finished at: $(date +"%T")"
