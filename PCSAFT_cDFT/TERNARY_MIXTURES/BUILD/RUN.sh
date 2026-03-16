#!/bin/bash

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

papermill VLE_IFT.ipynb VLE_IFT_output.ipynb