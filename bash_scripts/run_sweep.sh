#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
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
cmake --build build -j 64

# Dataset-ensemble Bayesian optimization sweep.
# --workers controls concurrent Optuna trials.
# --ensemble-workers controls concurrent dataset evaluations inside each trial.
# Keep workers * ensemble-workers <= 128 for this Slurm allocation.
# Pruning is off by default; add --pruning to enable Optuna median pruning.
uv run scripts/run_dataset_sweep.py \
  --config scripts/config.yaml \
  "${RUNS_ROOT_BASE_ARGS[@]}" \
  --datasets 16 \
  --trials-per-dataset 64 \
  --workers 8 \
  --ensemble-workers 8 \
  --seeds-start 1000 \
  --objective-metric track_efficiency \
  --objective-direction maximize \
  --ensemble-statistic mean \
  --trim-fraction 0.1 \
  --min-datasets-before-pruning 3 \
  --theta-max 0.30 0.85 \
  --angle-penalty 0.5 3.5 \
  --layer-radius-penalty 3.0 7.0 \
  --length-penalty 0.0 0.20 \
  --layer01-radial-tolerance 0.10 0.30 \
  --curvature-bonus 0.2 1.2 \
  --curvature-penalty 0.05 0.6 \
  --curvature-tolerance 0.02 0.18
