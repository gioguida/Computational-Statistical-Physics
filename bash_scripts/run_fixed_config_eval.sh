#!/bin/bash
#SBATCH --job-name=fixed-config-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=256M
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

mkdir -p logs .mplconfig

module load stack gcc cmake python

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PROJECT_ROOT}/.mplconfig"

MAX_WORKERS=48
WORKERS="${FIXED_WORKERS:-${SLURM_CPUS_PER_TASK:-48}}"
if (( WORKERS > MAX_WORKERS )); then
  WORKERS="$MAX_WORKERS"
fi
if (( WORKERS < 1 )); then
  WORKERS=1
fi

DATASETS="${FIXED_DATASETS:-96}"
DATASET_SEED_START="${FIXED_DATASET_SEED_START:-1000}"
ANNEAL_SEED_START="${FIXED_ANNEAL_SEED_START:-}"
OUTPUT_ROOT="${FIXED_OUTPUT_ROOT:-results/fixed_config_eval}"
SCRATCH_ROOT="${FIXED_SCRATCH_ROOT:-${TMPDIR:-$PROJECT_ROOT/results/tmp}/fixed_config_eval_${SLURM_JOB_ID:-local}}"
KEEP_RUN_ARTIFACTS="${FIXED_KEEP_RUN_ARTIFACTS:-0}"
VARY_ANNEAL_SEED="${FIXED_VARY_ANNEAL_SEED:-0}"

if command -v uv >/dev/null 2>&1; then
  PYTHON_RUN=(uv run python)
elif [ -x .venv/bin/python ]; then
  PYTHON_RUN=(.venv/bin/python)
else
  echo "Missing Python runner: install uv or create .venv/bin/python" >&2
  exit 1
fi

# Build once, then reuse the binaries across all worker threads.
cmake -S . -B build
cmake --build build -j "$WORKERS"

CMD=(
  "${PYTHON_RUN[@]}"
  scripts/evaluate_fixed_config.py
  --config scripts/config.yaml
  --datasets "$DATASETS"
  --workers "$WORKERS"
  --dataset-seed-start "$DATASET_SEED_START"
  --output-root "$OUTPUT_ROOT"
  --scratch-root "$SCRATCH_ROOT"
)

if [[ -n "$ANNEAL_SEED_START" ]]; then
  CMD+=(--anneal-seed-start "$ANNEAL_SEED_START")
fi
if (( KEEP_RUN_ARTIFACTS != 0 )); then
  CMD+=(--keep-run-artifacts)
fi
if (( VARY_ANNEAL_SEED != 0 )); then
  CMD+=(--vary-anneal-seed)
fi

printf 'Running:'
for token in "${CMD[@]}"; do
  printf ' %q' "$token"
done
printf '\n'

"${CMD[@]}"
