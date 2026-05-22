#!/bin/bash
# Sequential local runner: executes VLE_IFT_V4.ipynb once per composition in
# CSV_feeds/Combined_compositions.csv. Use this when SLURM is not available.
# Outputs go to LOCAL_RUN/<index>/ to mirror the SLURM layout.

set -euo pipefail

cd "$(dirname "$0")"

NOTEBOOK="VLE_IFT_V4.ipynb"
COMPOSITIONS_CSV="CSV_feeds/Combined_compositions.csv"

if [ ! -f "${NOTEBOOK}" ]; then
    echo "Error: ${NOTEBOOK} not found in $(pwd)"
    exit 1
fi
if [ ! -f "${COMPOSITIONS_CSV}" ]; then
    echo "Error: ${COMPOSITIONS_CSV} not found in $(pwd)"
    exit 1
fi

N_COMPOSITIONS="$(python3 -c "import csv; print(sum(1 for _ in open('${COMPOSITIONS_CSV}', newline='')) - 1)")"
if [ "${N_COMPOSITIONS}" -le 0 ]; then
    echo "Error: no compositions found in ${COMPOSITIONS_CSV}"
    exit 1
fi

if [ -f /home/darshan/A6/py_A6/bin/activate ]; then
    source /home/darshan/A6/py_A6/bin/activate
elif [ -f /home/darshan/A6/.venv/bin/activate ]; then
    source /home/darshan/A6/.venv/bin/activate
fi
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"
export MPLBACKEND=Agg

# Let FEOS (Rayon) and BLAS/LAPACK use multiple cores per notebook run.
# Override with THREADS=N ./RUN_LOCAL.sh ...
THREADS="${THREADS:-4}"
export RAYON_NUM_THREADS="${THREADS}"
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS="${THREADS}"
export NUMEXPR_NUM_THREADS="${THREADS}"
echo "Threads   : ${THREADS} (RAYON/OMP/MKL/OPENBLAS/NUMEXPR)"

command -v papermill >/dev/null 2>&1 || { echo "Error: papermill not found in PATH"; exit 1; }

# Optional: only run a subset by passing indices, e.g. ./RUN_LOCAL.sh 0 2
if [ "$#" -gt 0 ]; then
    INDICES=("$@")
else
    INDICES=()
    for ((i=0; i<N_COMPOSITIONS; i++)); do INDICES+=("$i"); done
fi

echo "Running ${#INDICES[@]} composition(s) sequentially from ${COMPOSITIONS_CSV}"
echo "Python    : $(which python)"
echo "Papermill : $(which papermill)"

for TASK_ID in "${INDICES[@]}"; do
    if [ "${TASK_ID}" -ge "${N_COMPOSITIONS}" ] || [ "${TASK_ID}" -lt 0 ]; then
        echo "Skip: index ${TASK_ID} out of range 0-$((N_COMPOSITIONS - 1))"
        continue
    fi

    OUTPUT_DIR="LOCAL_RUN/${TASK_ID}"
    CSV_DIR="${OUTPUT_DIR}/CSV"
    PLOT_DIR="${OUTPUT_DIR}/PLOTS"
    mkdir -p "${CSV_DIR}" "${PLOT_DIR}"

    echo ""
    echo "=== FEED_INDEX=${TASK_ID}  $(date '+%F %T') ==="
    papermill "${NOTEBOOK}" "${OUTPUT_DIR}/VLE_IFT_V4_${TASK_ID}.ipynb" \
        -p FEED_INDEX "${TASK_ID}" \
        -p SLURM_RUN False \
        -p verbose False \
        -p CSV_FOLDER "${CSV_DIR}" \
        -p PLOT_FOLDER "${PLOT_DIR}" \
        -p CRITICAL_REGION_ENHANCEMENTS False \
        -p IFT_CONTOUR_PLOT_ENHANCEMENTS False
done

echo ""
echo "All local runs finished at: $(date '+%F %T')"
