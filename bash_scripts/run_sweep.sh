#!/bin/bash
#SBATCH --job-name=csp_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"

module load stack gcc cmake python

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Ensure Python deps are present for this user environment.
uv sync

# Build once
cmake -S . -B build
cmake --build build -j 64

# Bayesian optimization sweep
uv run scripts/run_dataset_sweep.py \
  --config scripts/config.yaml \
  --datasets 8 \
  --trials-per-dataset 64 \
  --workers 64 \
  --seeds-start 1000 \
  --theta-max 0.25 0.45 \
  --angle-penalty 1.0 4.0 \
  --layer-radius-penalty 2.0 8.0 \
  --length-penalty 0.2 1.0 \
  --layer01-radial-tolerance 0.15 0.30
