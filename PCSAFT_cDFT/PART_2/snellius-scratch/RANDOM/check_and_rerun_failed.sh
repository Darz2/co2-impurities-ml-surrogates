#!/bin/bash

#SBATCH -J RANDOM_CHECK_RERUN
#SBATCH -t 04:00:00
#SBATCH -p gpu_h100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /scratch-shared/draju/PART_2/RANDOM/slurm-check-rerun-%j.out

# Checks all 160 expected output notebooks for valid CV data, then reruns
# any that are missing or failed — sequentially on a single GPU.

set -uo pipefail

echo "Job started at: $(date +"%T")"

module load 2025
module load MPICH/4.3.0-GCC-14.2.0-CUDA-12.8.0
module load Python/3.13.1-GCCcore-14.2.0

cd /scratch-shared/draju/PART_2/RANDOM
source /gpfs/home6/draju/A6/.A6/bin/activate

echo "CUDA Version: ${EBVERSIONCUDA}"

DATA_ROOT="/scratch-shared/draju/PART_2/RANDOM/COMBINED"
OUTPUT_ROOT="/scratch-shared/draju/PART_2/RANDOM/OUTPUTS"
SEED_BUBBLE=50015
SEED_GAMMA=50005

SIZES=(N025 N050 N075 N100)
TRIALS=()
for i in $(seq 0 19); do TRIALS+=( "$(printf 'trial_%02d' "$i")" ); done

# ── Python checker: returns 0 if notebook has parseable 5-fold CV output ──────
CHECKER=$(mktemp /tmp/check_cv_XXXXXX.py)
cat > "$CHECKER" << 'PYEOF'
import sys, json, re
import numpy as np

nb_path = sys.argv[1]
try:
    with open(nb_path) as f:
        nb = json.load(f)
except Exception:
    sys.exit(1)

arr_re = re.compile(r'\[([\d\.\s\-eE]+)\]')
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    if 'cross_validate' not in ''.join(cell['source']):
        continue
    for out in cell.get('outputs', []):
        text = out.get('text', out.get('data', {}).get('text/plain', ''))
        if isinstance(text, list):
            text = ''.join(text)
        for line in text.splitlines():
            m = arr_re.search(line)
            if m:
                arr = np.fromstring(m.group(1), sep=' ')
                if len(arr) == 5:
                    sys.exit(0)  # valid CV data found
sys.exit(1)
PYEOF

check_cv() { python3 "$CHECKER" "$1" 2>/dev/null; }

# ── Phase 1: Check ────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 1: Checking all 160 notebooks ==="

ok_count=0
fail_count=0
declare -a FAILED_JOBS=()

for trial in "${TRIALS[@]}"; do
    for size in "${SIZES[@]}"; do
        out_dir="${OUTPUT_ROOT}/${size}"
        for target in Bubble Gamma; do
            if [[ "$target" == "Bubble" ]]; then
                nb_out="${out_dir}/TabPFNBubble_${size}_${trial}_output.ipynb"
            else
                nb_out="${out_dir}/TabPFNGamma_${size}_${trial}_output.ipynb"
            fi

            if check_cv "$nb_out"; then
                ok_count=$(( ok_count + 1 ))
            else
                fail_count=$(( fail_count + 1 ))
                FAILED_JOBS+=("${size}:${trial}:${target}")
                echo "  FAIL  ${size}/${trial}/${target}"
            fi
        done
    done
done

echo ""
echo "Check complete — OK: ${ok_count}   FAIL: ${fail_count}"

if [[ ${#FAILED_JOBS[@]} -eq 0 ]]; then
    echo "All notebooks passed. Nothing to rerun."
    rm -f "$CHECKER"
    exit 0
fi

# ── Phase 2: Rerun failures ───────────────────────────────────────────────────
echo ""
echo "=== Phase 2: Rerunning ${fail_count} failed notebooks ==="

rerun_idx=0
for job in "${FAILED_JOBS[@]}"; do
    IFS=':' read -r size trial target <<< "$job"
    out_dir="${OUTPUT_ROOT}/${size}"
    mkdir -p "${out_dir}/Bubble" "${out_dir}/Gamma"
    rerun_idx=$(( rerun_idx + 1 ))

    if [[ "$target" == "Bubble" ]]; then
        nb_out="${out_dir}/TabPFNBubble_${size}_${trial}_output.ipynb"
        log="${out_dir}/TabPFNBubble_${size}_${trial}.log"
        nb_tmpl="TabPFNBubble_Calc.ipynb"
        seed="${SEED_BUBBLE}"
        plot_dir="${out_dir}/Bubble"
    else
        nb_out="${out_dir}/TabPFNGamma_${size}_${trial}_output.ipynb"
        log="${out_dir}/TabPFNGamma_${size}_${trial}.log"
        nb_tmpl="TabPFNGamma_Calc.ipynb"
        seed="${SEED_GAMMA}"
        plot_dir="${out_dir}/Gamma"
    fi

    echo "[${rerun_idx}/${fail_count}] ${size}/${trial}/${target} — $(date +"%T")"
    CUDA_VISIBLE_DEVICES=0 papermill "$nb_tmpl" "$nb_out" \
        -p PLOT_FOLDER "$plot_dir" \
        -p TEST_ROWS   None \
        -p SEED        "$seed" \
        -p DATA_PATH   "${DATA_ROOT}/${size}/${trial}.csv" \
        > "$log" 2>&1 \
        || echo "  ERROR during rerun — check $log"
    echo "       Done at $(date +"%T")"
done

rm -f "$CHECKER"
echo ""
echo "All reruns finished at: $(date +"%T")"
