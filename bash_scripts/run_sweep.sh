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

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Build once
cmake -S . -B build
cmake --build build -j 64

python3 scripts/run_dataset_sweep.py \
  --config scripts/config.yaml \
  --datasets 8 \
  --workers 64 \
  --seeds-start 1000 \
  --theta-max 0.25 0.35 0.45 \
  --angle-penalty 1.0 2.0 4.0 \
  --layer-radius-penalty 2.0 5.0 8.0 \
  --length-penalty 0.2 0.5 1.0 \
  --layer01-radial-tolerance 0.15 0.23 0.30
