#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

mkdir -p logs

module load stack gcc cmake python

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MAX_WORKERS=48
WORKERS="${SWEEP_WORKERS:-${SLURM_CPUS_PER_TASK:-48}}"
if (( WORKERS > MAX_WORKERS )); then
  WORKERS="$MAX_WORKERS"
fi

DATASETS="${SWEEP_DATASETS:-12}"
TRIALS_PER_DATASET="${SWEEP_TRIALS_PER_DATASET:-96}"
SEEDS_START="${SWEEP_SEEDS_START:-1000}"
OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-results/sweeps}"

# Build once before launching the Bayesian sweep.
cmake -S . -B build
cmake --build build -j "$WORKERS"

# Search space is centered on the current best region, with extra room for the
# new soft angular/radial cliffs to support stronger full penalties.
uv run python scripts/run_dataset_sweep.py \
  --config scripts/config.yaml \
  --datasets "$DATASETS" \
  --trials-per-dataset "$TRIALS_PER_DATASET" \
  --workers "$WORKERS" \
  --seeds-start "$SEEDS_START" \
  --output-root "$OUTPUT_ROOT" \
  --theta-max 0.28 0.42 \
  --angle-penalty 4.5 8.5 \
  --layer-radius-penalty 4.5 8.5 \
  --length-penalty 0.20 0.65 \
  --layer01-radial-tolerance 0.12 0.24 \
  --anneal-t-max 4.0 8.0 \
  --max-fake-rate 0.20 \
  --max-bifurcations 0 \
  --log-every-steps 200 \
  --checkpoint-every-steps 600
