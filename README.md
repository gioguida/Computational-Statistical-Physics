# Particle Track Reconstruction as a Spin-Glass Optimization Problem

This README is a practical runbook to execute and verify the project, both locally and on the Euler cluster.

## Project Overview

This project studies particle track reconstruction in a simplified detector setting by casting it as a combinatorial optimization problem.  
Hits on concentric detector layers are connected into candidate segments, and those segments are encoded in a spin-glass / Ising-style energy model.

The objective is to select a globally consistent set of segments that forms plausible tracks while discouraging conflicts such as forks and merges.  
The optimization is solved with simulated annealing in C++, while Python orchestrates data generation, experiment management, and metrics/plots.

In practical terms, the workflow is:

1. Generate synthetic detector hits (clean ground truth + noisy training hits).
2. Build candidate segments and pairwise couplings between segments.
3. Minimize the resulting energy with simulated annealing.
4. Compare reconstructed tracks against ground truth with precision/recall- and efficiency-based metrics.

The key modeling idea is that compatible segment pairs receive favorable couplings, while incompatible combinations (for example multiple segment continuations through the same hit) are penalized.  
A low-energy state should therefore correspond to a physically coherent set of tracks.

## Method At A Glance

- `src/hits/`: synthetic data generation for layered detector hits.
- `src/interaction/`: construction of segment graph and sparse coupling matrix.
- `src/annealing/`: C++ simulated annealing solver over binary segment-selection variables.
- `src/eval/` and `src/plotting/`: quantitative evaluation and visual diagnostics.

## What To Run First (Fast Verification)

From the project root:

```bash
uv run python scripts/control_panel.py
```

If this command completes, the core pipeline works end-to-end (build, data generation, interaction matrix, annealing, plots, metrics).

## Prerequisites

- Python >= 3.12
- CMake >= 3.15
- C++17 compiler
- `uv` available (`pip install uv`)

## Repository Entry Points

- `scripts/control_panel.py`: end-to-end single run (local verification).
- `scripts/run_dataset_sweep.py`: Bayesian hyperparameter sweep across many datasets.
- `scripts/evaluate_fixed_config.py`: large fixed-parameter evaluation across many datasets.
- `bash_scripts/run_sweep.sh`: Slurm script for sweep on cluster.
- `bash_scripts/run_fixed_eval.sh`: Slurm script for fixed evaluation on cluster.

## Local End-to-End Verification

1. Build + run one full pipeline:

```bash
uv run python scripts/control_panel.py
```

2. Confirm a new run folder was created in `results/runs/<run_id>/` with:

- `interaction/segments.csv`
- `interaction/J_edges.csv`
- `annealing/final_state.csv`
- `annealing/energy_trace.csv`
- `annealing/annealing_meta.json`

3. Confirm post-processing exists in the same run folder (metrics/plots outputs).

## Cluster: Hyperparameter Sweep (Slurm)

### Submit

```bash
sbatch bash_scripts/run_sweep.sh
```

### Defaults used by the script

- `--cpus-per-task=48`
- CMake build with `-j 48`
- Sweep parallelism:
  - `--workers 6`
  - `--ensemble-workers 8`
  - Total max concurrent pipelines: `6 x 8 = 48`

### Output location

- Main outputs: `results/sweeps/<timestamp>/`
- If `$SCRATCH` is set, run artifacts are redirected there via `--runs-root-base`.

### Files to verify

In `results/sweeps/<timestamp>/` check:

- `summary_trials.csv`
- `best_per_dataset.csv`
- `ensemble_trials.csv`
- `ensemble_best.csv`
- `manifest.json`

## Cluster: Fixed-Config Evaluation (Slurm)

### Submit

```bash
sbatch bash_scripts/run_fixed_eval.sh
```

### Defaults used by the script

- `--cpus-per-task=48`
- Worker cap:
  - `MAX_WORKERS=48`
  - `WORKERS=48`
- Default dataset count: `128`

### Output location

- Main outputs: `results/fixed_config_eval/<timestamp>/`
- Temporary worker artifacts are placed in `$SCRATCH` if available, else under tmp.

### Files to verify

In `results/fixed_config_eval/<timestamp>/` check:

- `summary_trials.csv`
- `ensemble_trials.csv`
- `per_dataset_metrics.csv`
- `aggregate_metrics.csv`
- `summary.json`
- `summary_report.md`

## How To Assess Correctness Quickly

For each run mode (local / sweep / fixed eval), verify:

1. Job/process exits successfully (local command return code 0, or Slurm job `COMPLETED`).
2. Expected output folder is created with the files listed above.
3. CSV outputs are non-empty.
4. No missing binaries error (`run_interaction`, `run_annealing`).
5. No failed dataset rows in summary tables for fixed evaluation (`state != COMPLETE` should be absent or minimal and explainable).

## Slurm Logs

Both Slurm scripts write logs to:

- `logs/<job-name>-<job-id>.out`
- `logs/<job-name>-<job-id>.err`

These are the first place to check for build/runtime failures.

## GitHub Reproducibility

This repository can be validated from a fresh clone with the same commands in this README.

```bash
git clone <your-repo-url>
cd <repo-folder>
uv run python scripts/control_panel.py
```

To confirm the code works after cloning:

1. The command completes without errors.
2. A new folder appears under `results/runs/<run_id>/`.
3. The expected artifacts are present (`segments.csv`, `J_edges.csv`, `final_state.csv`, `energy_trace.csv`, metrics/plot outputs).

For larger experiments on cluster, submit:

```bash
sbatch bash_scripts/run_sweep.sh
sbatch bash_scripts/run_fixed_eval.sh
```

Then verify the output files listed in the sweep and fixed-evaluation sections above.
