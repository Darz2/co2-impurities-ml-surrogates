#!/bin/bash

#SBATCH --job-name=cDFT_Binary-(CO2-CH4)
#SBATCH --partition=parallel-short
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

start_time=$(date +"%T")
echo "Job started at: $start_time"

source  /home/darshan/A6/py_A6/bin/activate
export PATH=$HOME/texlive/2025/bin/x86_64-linux:$PATH

which python
which latex

papermill CO2_CH4.ipynb OUTPUT/CO2_CH4_output.ipynb

end_time=$(date +"%T")
echo "Job ended at: $end_time"
total_seconds=$(( $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) ))
total_minutes=$(echo "scale=2; $total_seconds/60" | bc)
echo "Total time: $total_minutes minutes"