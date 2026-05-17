#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --constraint=EPYC_7763
#SBATCH --mem-per-cpu=512
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"

mkdir -p logs

module load stack gcc cmake python

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUNS_ROOT_BASE_ARGS=()
if [[ -n "${SCRATCH:-}" ]]; then
  RUNS_ROOT_BASE_ARGS+=(--runs-root-base "$SCRATCH/CSP/sweeps")
fi

# Build once
cmake -S . -B build
cmake --build build -j 48

# Dataset-ensemble Bayesian optimization sweep.
# --workers controls concurrent Optuna trials.
# --ensemble-workers controls concurrent dataset evaluations inside each trial.
# Keep workers * ensemble-workers <= 48 for this Slurm allocation.
# Pruning is off by default; add --pruning to enable Optuna median pruning.
uv run scripts/run_dataset_sweep.py \
  --config scripts/config.yaml \
  "${RUNS_ROOT_BASE_ARGS[@]}" \
  --datasets 16 \
  --trials-per-dataset 80 \
  --workers 6 \
  --ensemble-workers 8 \
  --seeds-start 1000 \
  --objective-metric track_efficiency \
  --objective-direction maximize \
  --ensemble-statistic mean \
  --trim-fraction 0.1 \
  --trial-timeout 600 \
  --head-starts bash_scripts/head_starts.json \
  --theta-max 0.25 0.65 \
  --angle-penalty 1.0 8.0 \
  --layer-radius-penalty 0.0 20.0 \
  --length-penalty 0.0 0.35 \
  --layer01-radial-tolerance 0.10 0.40 \
  --curvature-bonus 0.0 2.0 \
  --curvature-penalty 0.0 0.5 \
  --curvature-tolerance 0.02 0.15 \
  --t-max 3.0 10.0 \
  --n-steps 300 1000 \
  --eq-sweeps 80 300 \
  --merge-penalty 5.0 20.0 \
  --fork-penalty 5.0 20.0
