#!/bin/bash
#SBATCH --job-name=SR_par_mixtures
#SBATCH --partition=highmem
#SBATCH --nodelist=c109
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-SR_par-%j.out

# ============================================================
#  Parallel pipeline for the SR mixtures study on ONE c109 node.
#
#  SymbolicRegression's multithreading does NOT scale to 256 threads
#  (hall-of-fame / migration contention), so instead of one 256-thread
#  job we run FOUR independent searches concurrently, each with a
#  moderate, well-scaling thread count:
#
#      job 1: V1 production   (Mixture.jl)
#      job 2: V2 production   (Mixture_V2.jl)
#      job 3: CV  V1          (SR_cross_validation.jl V1)
#      job 4: CV  V2          (SR_cross_validation.jl V2)
#
#  All four run in the background and we `wait` for them; the parity /
#  residual plots run afterwards (they need the production CSVs).
#
#  Submit:  sbatch RUN_PARALLEL_SR.sh
# ============================================================

set -euo pipefail

# Threads per concurrent job. 4 x 48 = 192 cores (<= 256), each job in a
# regime where SR still scales well. Override: sbatch --export=ALL,THREADS_PER_JOB=56 ...
THREADS_PER_JOB=${THREADS_PER_JOB:-48}
export SR_CV_NITERATIONS=${SR_CV_NITERATIONS:-200}

echo "=================================================="
echo "Job started at  : $(date +'%F %T')"
echo "Node            : $(hostname)"
echo "CPUs allocated  : ${SLURM_CPUS_PER_TASK}"
echo "Threads per job : ${THREADS_PER_JOB}  (x4 concurrent = $((THREADS_PER_JOB*4)))"
echo "SR_CV_NITERATIONS: ${SR_CV_NITERATIONS}"
echo "=================================================="

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ── Environment ───────────────────────────────────────────────────────────────
export PATH="$HOME/.juliaup/bin:$PATH"
export PATH="$HOME/Software/texlive/2025/bin/x86_64-linux:$PATH"
source /home/darshan/A6/py_A6/bin/activate
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p logs

# ── Launch the four independent searches concurrently ─────────────────────────
echo ">>> [$(date +'%T')] launching 4 parallel searches ..."

JULIA_NUM_THREADS=${THREADS_PER_JOB} julia Mixture.jl              > logs/v1_prod.log    2>&1 &
PID_V1P=$!
JULIA_NUM_THREADS=${THREADS_PER_JOB} julia Mixture_V2.jl           > logs/v2_prod.log    2>&1 &
PID_V2P=$!
JULIA_NUM_THREADS=${THREADS_PER_JOB} julia SR_cross_validation.jl V1 > logs/v1_cv.log    2>&1 &
PID_V1CV=$!
JULIA_NUM_THREADS=${THREADS_PER_JOB} julia SR_cross_validation.jl V2 > logs/v2_cv.log    2>&1 &
PID_V2CV=$!

echo "    PIDs: V1prod=${PID_V1P} V2prod=${PID_V2P} V1cv=${PID_V1CV} V2cv=${PID_V2CV}"

# ── Wait for all, capturing per-job exit status ───────────────────────────────
rc=0
wait ${PID_V1P}  || { echo "!! V1 production FAILED (see logs/v1_prod.log)"; rc=1; }
echo ">>> [$(date +'%T')] V1 production done"
wait ${PID_V2P}  || { echo "!! V2 production FAILED (see logs/v2_prod.log)"; rc=1; }
echo ">>> [$(date +'%T')] V2 production done"
wait ${PID_V1CV} || { echo "!! V1 CV FAILED (see logs/v1_cv.log)"; rc=1; }
echo ">>> [$(date +'%T')] V1 CV done"
wait ${PID_V2CV} || { echo "!! V2 CV FAILED (see logs/v2_cv.log)"; rc=1; }
echo ">>> [$(date +'%T')] V2 CV done"

# ── Plots (need both production CSVs; run only if production succeeded) ────────
if [ "${rc}" -eq 0 ]; then
    echo ">>> [$(date +'%T')] post-processing plots (make_sr_plots.py)"
    python make_sr_plots.py > logs/plots.log 2>&1 || { echo "!! plotting FAILED (see logs/plots.log)"; rc=1; }
else
    echo "!! skipping plots because a search failed"
fi

echo "=================================================="
echo "All steps finished at: $(date +'%F %T')  (rc=${rc})"
echo "Outputs:"
echo "  SR_MIXTURES_OUTPUTS/        (V1 equation, metrics, plots, CSVs)"
echo "  SR_MIXTURES_OUTPUTS_V2/     (V2 equation, metrics, plots, CSVs)"
echo "  SR_CV_OUTPUTS/SR_CV_metrics_V1.json, SR_CV_metrics_V2.json"
echo "  logs/                       (per-job stdout)"
echo "Next: update + recompile SR_Mixtures_Report.tex"
echo "=================================================="
exit ${rc}
