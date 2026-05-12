#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import json
import math
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hits.data_gen import DataConfig
from src.hits.hits import Detector
from src.plotting.metrics import visualize_metrics


ENSEMBLE_STATISTICS = (
    "mean",
    "median",
    "min",
    "max",
    "trim_mean",
    "mean_minus_std",
    "mean_minus_2std",
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be mapping: {path}")
    return cfg


def generate_dataset(cfg: DataConfig, seed: int, out_dir: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)

    legacy_state = np.random.get_state()
    try:
        np.random.seed(seed)
        experiment = Detector(cfg.detector_layers, cfg.n_particles, cfg.traj_radius_low, cfg.traj_radius_high)
        clean_hits = experiment.get_hits()
    finally:
        np.random.set_state(legacy_state)
    n_real_hits = len(clean_hits)

    noisy_hits = clean_hits.copy()
    noisy_hits["hit_x"] = noisy_hits["hit_x"] + rng.normal(0.0, cfg.sigma_res, len(noisy_hits))
    noisy_hits["hit_y"] = noisy_hits["hit_y"] + rng.normal(0.0, cfg.sigma_res, len(noisy_hits))
    noisy_hits.insert(0, "hit_id", np.arange(n_real_hits, dtype=int))

    fake_rows: list[dict[str, Any]] = []
    for layer_id, layer_radius in enumerate(cfg.detector_layers):
        n_fake = int(rng.poisson(lam=cfg.mean_fakes_per_layer))
        angles = rng.uniform(0.0, 2.0 * np.pi, n_fake)
        x_fake = layer_radius * np.cos(angles)
        y_fake = layer_radius * np.sin(angles)
        for i in range(n_fake):
            fake_rows.append(
                {
                    "track_id": -1,
                    "layer_id": layer_id,
                    "layer_radius": float(layer_radius),
                    "hit_x": float(x_fake[i]),
                    "hit_y": float(y_fake[i]),
                    "hit_phi": float(angles[i]),
                }
            )

    fake_hits = pd.DataFrame(fake_rows, columns=clean_hits.columns)
    fake_hits.insert(0, "hit_id", np.arange(n_real_hits, n_real_hits + len(fake_hits), dtype=int))
    all_hits = pd.concat([noisy_hits, fake_hits], ignore_index=True)
    all_hits = all_hits.sort_values(["layer_id", "track_id"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = out_dir / "ground_truth_hits.csv"
    train_path = out_dir / "training_hits.csv"

    ground_truth_hits = clean_hits.copy()
    ground_truth_hits.insert(0, "hit_id", np.arange(n_real_hits, dtype=int))
    ground_truth_hits.to_csv(gt_path, index=False)
    all_hits.drop(columns=["track_id"]).to_csv(train_path, index=False)
    return train_path, gt_path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def run_one_job(job: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(job["project_root"])
    build_dir = Path(job["build_dir"])
    run_dir = Path(job["run_dir"])
    inter_dir = run_dir / "interaction"
    ann_dir = run_dir / "annealing"
    inter_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    interaction_bin = build_dir / "run_interaction"
    annealing_bin = build_dir / "run_annealing"

    hits_csv = Path(job["hits_csv"])
    gt_csv = Path(job["gt_csv"])

    p = job["params"]
    ann = job["annealing_base"]

    run_cmd(
        [
            str(interaction_bin),
            "--hits-csv",
            str(hits_csv),
            "--out-dir",
            str(inter_dir),
            "--theta-max",
            str(p["theta_max"]),
            "--merge-penalty",
            str(job["merge_penalty"]),
            "--fork-penalty",
            str(job["fork_penalty"]),
            "--angle-penalty",
            str(p["angle_penalty"]),
        ],
        cwd=project_root,
    )

    run_cmd(
        [
            str(annealing_bin),
            "--hits-csv",
            str(hits_csv),
            "--segments-csv",
            str(inter_dir / "segments.csv"),
            "--edges-csv",
            str(inter_dir / "J_edges.csv"),
            "--out-dir",
            str(ann_dir),
            "--t-min",
            str(ann["t_min"]),
            "--t-max",
            str(ann["t_max"]),
            "--n-steps",
            str(ann["n_steps"]),
            "--toll",
            str(ann["toll"]),
            "--length-penalty",
            str(p["length_penalty"]),
            "--layer01-radial-penalty",
            str(p["layer01_radial_penalty"]),
            "--layer01-radial-tolerance",
            str(p["layer01_radial_tolerance"]),
            "--first-gap",
            str(job["first_gap"]),
            "--eq-sweeps",
            str(ann["eq_sweeps"]),
            "--log-every-steps",
            str(ann["log_every_steps"]),
            "--checkpoint-every-steps",
            str(ann["checkpoint_every_steps"]),
            "--cooling-schedule",
            str(ann["cooling_schedule"]),
            "--seed",
            str(job["anneal_seed"]),
            "--curvature-bonus",
            str(p["curvature_bonus"]),
            "--curvature-penalty",
            str(p["curvature_penalty"]),
            "--curvature-tolerance",
            str(p["curvature_tolerance"]),
        ],
        cwd=project_root,
    )

    metrics = visualize_metrics(
        {
            "project_root": str(project_root),
            "run_dir": str(run_dir),
            "training_hits_csv": str(hits_csv),
            "ground_truth_csv": str(gt_csv),
            "n_layers": int(job["n_layers"]),
            "merge_penalty": float(job["merge_penalty"]),
            "fork_penalty": float(job["fork_penalty"]),
            "angle_penalty": float(p["angle_penalty"]),
        }
    )

    row = {
        "run_id": run_dir.name,
        **p,
        "segment_precision": float(metrics.get("precision", 0.0)),
        "segment_recall": float(metrics.get("TPR", 0.0)),
        "track_efficiency": float(metrics.get("track_efficiency", 0.0)),
        "track_fake_rate": float(metrics.get("fake_rate", 0.0)),
        "n_bifurcations": int(metrics.get("n_bifurcations", 0)),
    }
    for key, value in metrics.items():
        if key not in row:
            row[key] = value
    return row


def _bounds(values: list[float], name: str) -> tuple[float, float]:
    if not values:
        raise ValueError(f"Missing values for {name}")
    lo = min(values)
    hi = max(values)
    return (float(lo), float(hi))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset-ensemble Bayesian optimization with Optuna")
    parser.add_argument("--config", default="scripts/config.yaml")
    parser.add_argument("--datasets", type=int, default=8)
    parser.add_argument("--trials-per-dataset", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--ensemble-workers", type=int, default=8)
    parser.add_argument("--seeds-start", type=int, default=1000)
    parser.add_argument("--output-root", default="results/sweeps")
    parser.add_argument("--runs-root-base", default=None)
    parser.add_argument("--theta-max", nargs="+", type=float, required=True)
    parser.add_argument("--angle-penalty", nargs="+", type=float, required=True)
    parser.add_argument("--layer01-radial-penalty", nargs="+", type=float, default=[0.0, 15.0])
    parser.add_argument("--length-penalty", nargs="+", type=float, required=True)
    parser.add_argument("--layer01-radial-tolerance", nargs="+", type=float, required=True)
    parser.add_argument("--curvature-bonus", nargs="+", type=float, required=True)
    parser.add_argument("--curvature-penalty", nargs="+", type=float, required=True)
    parser.add_argument("--curvature-tolerance", nargs="+", type=float, required=True)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--max-fake-rate", type=float, default=None)
    parser.add_argument("--max-bifurcations", type=int, default=None)
    parser.add_argument("--objective-metric", default="track_efficiency")
    parser.add_argument("--objective-direction", choices=("maximize", "minimize"), default="maximize")
    parser.add_argument("--ensemble-statistic", choices=ENSEMBLE_STATISTICS, default="mean")
    parser.add_argument("--trim-fraction", type=float, default=0.1)
    parser.add_argument("--pruning", action="store_true")
    parser.add_argument("--min-datasets-before-pruning", type=int, default=3)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.datasets <= 0:
        raise ValueError("--datasets must be positive")
    if args.trials_per_dataset <= 0:
        raise ValueError("--trials-per-dataset must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.ensemble_workers <= 0:
        raise ValueError("--ensemble-workers must be positive")
    if not (0.0 < args.trim_fraction < 0.5):
        raise ValueError("--trim-fraction must be in (0, 0.5)")
    if args.min_datasets_before_pruning < 1:
        raise ValueError("--min-datasets-before-pruning must be at least 1")
    if args.min_datasets_before_pruning >= args.datasets:
        raise ValueError("--min-datasets-before-pruning must be less than --datasets")


def finite_values(values: list[float]) -> np.ndarray:
    return np.array([value for value in values if math.isfinite(value)], dtype=float)


def aggregate_values(values: list[float], statistic: str, trim_fraction: float) -> float:
    arr = finite_values(values)
    if len(arr) == 0:
        return float("nan")
    if statistic == "mean":
        return float(arr.mean())
    if statistic == "median":
        return float(np.median(arr))
    if statistic == "min":
        return float(arr.min())
    if statistic == "max":
        return float(arr.max())
    if statistic == "mean_minus_std":
        return float(arr.mean() - arr.std(ddof=1)) if len(arr) > 1 else float(arr.mean())
    if statistic == "mean_minus_2std":
        return float(arr.mean() - 2.0 * arr.std(ddof=1)) if len(arr) > 1 else float(arr.mean())
    if statistic == "trim_mean":
        sorted_arr = np.sort(arr)
        trim_count = int(math.floor(len(sorted_arr) * trim_fraction))
        if trim_count > 0 and 2 * trim_count < len(sorted_arr):
            sorted_arr = sorted_arr[trim_count:-trim_count]
        return float(sorted_arr.mean())
    raise ValueError(f"Unknown ensemble statistic: {statistic}")


def ensemble_summary(values: list[float]) -> dict[str, float]:
    arr = finite_values(values)
    if len(arr) == 0:
        return {
            "ensemble_mean": float("nan"),
            "ensemble_median": float("nan"),
            "ensemble_std": float("nan"),
            "ensemble_min": float("nan"),
            "ensemble_max": float("nan"),
        }
    return {
        "ensemble_mean": float(arr.mean()),
        "ensemble_median": float(np.median(arr)),
        "ensemble_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "ensemble_min": float(arr.min()),
        "ensemble_max": float(arr.max()),
    }


def failed_objective(direction: str) -> float:
    return float("inf") if direction == "minimize" else float("-inf")


def main() -> int:
    args = parse_args()
    validate_args(args)
    cfg = load_yaml((PROJECT_ROOT / args.config).resolve())

    build_dir = (PROJECT_ROOT / str(cfg.get("build", {}).get("build_dir", "build"))).resolve()
    interaction_bin = build_dir / "run_interaction"
    annealing_bin = build_dir / "run_annealing"
    if not interaction_bin.exists() or not annealing_bin.exists():
        raise FileNotFoundError("Missing binaries in build dir. Run cmake build first.")

    gen_cfg = DataConfig.from_yaml((PROJECT_ROOT / args.config).resolve())
    detector_layers = gen_cfg.detector_layers
    if len(detector_layers) < 2:
        raise ValueError("Need at least 2 detector layers")
    first_gap = abs(float(detector_layers[1]) - float(detector_layers[0]))

    inter_cfg = cfg.get("interaction", {})
    ann_cfg = cfg.get("annealing", {})

    merge_penalty = float(inter_cfg.get("merge_penalty", 10.0))
    fork_penalty = float(inter_cfg.get("fork_penalty", 10.0))

    annealing_base = {
        "t_min": float(ann_cfg.get("t_min", 1e-3)),
        "t_max": float(ann_cfg.get("t_max", 2.0)),
        "n_steps": int(ann_cfg.get("n_steps", 300)),
        "toll": float(ann_cfg.get("toll", 1e-6)),
        "eq_sweeps": int(ann_cfg.get("eq_sweeps", 100)),
        "log_every_steps": int(ann_cfg.get("log_every_steps", 10)),
        "checkpoint_every_steps": int(ann_cfg.get("checkpoint_every_steps", 10)),
        "cooling_schedule": str(ann_cfg.get("cooling_schedule", "geometric")),
    }

    theta_lo, theta_hi = _bounds(args.theta_max, "theta_max")
    angle_lo, angle_hi = _bounds(args.angle_penalty, "angle_penalty")
    layer_radius_lo, layer_radius_hi = _bounds(args.layer01_radial_penalty, "layer01_radial_penalty")
    length_lo, length_hi = _bounds(args.length_penalty, "length_penalty")
    tol_lo, tol_hi = _bounds(args.layer01_radial_tolerance, "layer01_radial_tolerance")
    curv_bonus_lo, curv_bonus_hi = _bounds(args.curvature_bonus, "curvature_bonus")
    curv_penalty_lo, curv_penalty_hi = _bounds(args.curvature_penalty, "curvature_penalty")
    curv_tol_lo, curv_tol_hi = _bounds(args.curvature_tolerance, "curvature_tolerance")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = (PROJECT_ROOT / args.output_root / stamp).resolve()
    datasets_root = sweep_root / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)
    if args.runs_root_base is not None:
        runs_root = (Path(args.runs_root_base).expanduser().resolve() / stamp / "runs").resolve()
    else:
        runs_root = sweep_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    total_trials = int(args.trials_per_dataset)
    print(f"Sweep root: {sweep_root}")
    print(f"Datasets: {args.datasets}")
    print(f"Trials per dataset: {args.trials_per_dataset}")
    print(
        (
            f"Parallelism: {args.workers} trials × {args.ensemble_workers} ensemble workers "
            f"= {args.workers * args.ensemble_workers} max concurrent pipelines"
        ),
        flush=True,
    )
    print(
        (
            f"Objective: {args.objective_direction} {args.ensemble_statistic} "
            f"of {args.objective_metric} over {args.datasets} datasets"
        ),
        flush=True,
    )

    datasets: list[dict[str, Any]] = []
    for d in range(args.datasets):
        dataset_id = f"ds_{d:03d}"
        dataset_dir = datasets_root / dataset_id
        dataset_seed = args.seeds_start + d
        train_csv, gt_csv = generate_dataset(gen_cfg, dataset_seed, dataset_dir)
        datasets.append(
            {
                "dataset_id": dataset_id,
                "dataset_index": d,
                "dataset_seed": dataset_seed,
                "train_csv": str(train_csv),
                "gt_csv": str(gt_csv),
            }
        )

    print(f"Generated {len(datasets)} datasets upfront.", flush=True)

    all_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    rows_lock = threading.Lock()

    def evaluate_dataset(trial_number: int, params: dict[str, float], dataset: dict[str, Any]) -> dict[str, Any]:
        run_id = f"{dataset['dataset_id']}_trial_{trial_number:04d}"
        run_dir = runs_root / run_id
        base_row: dict[str, Any] = {
            "dataset_id": dataset["dataset_id"],
            "dataset_seed": dataset["dataset_seed"],
            "trial_number": trial_number,
            "run_id": run_id,
            **params,
        }
        try:
            row = run_one_job(
                {
                    "project_root": str(PROJECT_ROOT),
                    "build_dir": str(build_dir),
                    "run_dir": str(run_dir),
                    "hits_csv": dataset["train_csv"],
                    "gt_csv": dataset["gt_csv"],
                    "n_layers": len(detector_layers),
                    "first_gap": first_gap,
                    "merge_penalty": merge_penalty,
                    "fork_penalty": fork_penalty,
                    "annealing_base": annealing_base,
                    "anneal_seed": int(ann_cfg.get("seed", 42)),
                    "params": params,
                }
            )
            metric_value = float(row[args.objective_metric])
            pruned = False
            prune_reason = None
            if args.max_fake_rate is not None and row["track_fake_rate"] > args.max_fake_rate:
                pruned = True
                prune_reason = (
                    f"fake_rate {row['track_fake_rate']:.6f} exceeded threshold {args.max_fake_rate:.6f}"
                )
            if (
                not pruned
                and args.max_bifurcations is not None
                and int(row["n_bifurcations"]) > int(args.max_bifurcations)
            ):
                pruned = True
                prune_reason = (
                    f"n_bifurcations {int(row['n_bifurcations'])} exceeded threshold {int(args.max_bifurcations)}"
                )
            row.update(base_row)
            row["objective_metric"] = args.objective_metric
            row["objective_metric_value"] = metric_value
            row["objective_value"] = metric_value
            row["pruned"] = pruned
            row["prune_reason"] = prune_reason
            row["state"] = "PRUNED" if pruned else "COMPLETE"
            return row
        except Exception as exc:
            print(
                (
                    f"WARNING: trial {trial_number} dataset {dataset['dataset_id']} failed: "
                    f"{exc}"
                ),
                file=sys.stderr,
                flush=True,
            )
            return {
                **base_row,
                "objective_metric": args.objective_metric,
                "objective_metric_value": float("nan"),
                "objective_value": float("nan"),
                "pruned": False,
                "prune_reason": None,
                "state": "FAIL",
                "error": str(exc),
            }

    def objective(trial: optuna.Trial) -> float:
        params = {
            "theta_max": trial.suggest_float("theta_max", theta_lo, theta_hi),
            "angle_penalty": trial.suggest_float("angle_penalty", angle_lo, angle_hi),
            "layer01_radial_penalty": trial.suggest_float(
                "layer01_radial_penalty", layer_radius_lo, layer_radius_hi
            ),
            "length_penalty": trial.suggest_float("length_penalty", length_lo, length_hi),
            "layer01_radial_tolerance": trial.suggest_float(
                "layer01_radial_tolerance", tol_lo, tol_hi
            ),
            "curvature_bonus": trial.suggest_float("curvature_bonus", curv_bonus_lo, curv_bonus_hi),
            "curvature_penalty": trial.suggest_float(
                "curvature_penalty", curv_penalty_lo, curv_penalty_hi
            ),
            "curvature_tolerance": trial.suggest_float(
                "curvature_tolerance", curv_tol_lo, curv_tol_hi
            ),
        }

        values_by_dataset: dict[str, float] = {}
        dataset_rows: list[dict[str, Any]] = []
        state = "COMPLETE"
        prune_reason = None

        with futures.ThreadPoolExecutor(max_workers=args.ensemble_workers) as executor:
            submitted = {
                executor.submit(evaluate_dataset, trial.number, params, dataset): dataset
                for dataset in datasets
            }
            for completed_count, future in enumerate(futures.as_completed(submitted), start=1):
                row = future.result()
                dataset_rows.append(row)
                values_by_dataset[row["dataset_id"]] = float(row["objective_metric_value"])

                current_values = list(values_by_dataset.values())
                current_objective = aggregate_values(
                    current_values,
                    statistic=args.ensemble_statistic,
                    trim_fraction=args.trim_fraction,
                )
                if args.pruning and completed_count >= args.min_datasets_before_pruning:
                    if math.isfinite(current_objective):
                        trial.report(current_objective, step=completed_count)
                    if trial.should_prune():
                        state = "PRUNED"
                        prune_reason = f"pruned after {completed_count} datasets"
                        for pending in submitted:
                            pending.cancel()
                        break

        values = [values_by_dataset.get(dataset["dataset_id"], float("nan")) for dataset in datasets]
        objective_value = aggregate_values(
            values,
            statistic=args.ensemble_statistic,
            trim_fraction=args.trim_fraction,
        )
        if not math.isfinite(objective_value):
            objective_value = failed_objective(args.objective_direction)
            state = "FAIL"

        successful_count = int(len(finite_values(values)))

        summary = {
            "trial_number": trial.number,
            **params,
            **{f"metric_{dataset['dataset_id']}": values_by_dataset.get(dataset["dataset_id"], float("nan"))
               for dataset in datasets},
            **ensemble_summary(values),
            "objective_value": objective_value,
            "objective_metric": args.objective_metric,
            "objective_statistic": args.ensemble_statistic,
            "state": state,
            "successful_datasets": successful_count,
            "failed_datasets": len(datasets) - successful_count,
            "prune_reason": prune_reason,
        }

        with rows_lock:
            all_rows.extend(dataset_rows)
            ensemble_rows.append(summary)

        trial.set_user_attr("objective_metric", args.objective_metric)
        trial.set_user_attr("objective_statistic", args.ensemble_statistic)
        trial.set_user_attr("successful_datasets", successful_count)
        trial.set_user_attr("failed_datasets", len(datasets) - successful_count)
        for key in ("ensemble_mean", "ensemble_median", "ensemble_std", "ensemble_min", "ensemble_max"):
            trial.set_user_attr(key, summary[key])

        if state == "PRUNED":
            raise optuna.TrialPruned(prune_reason)
        return float(objective_value)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    pruner: optuna.pruners.BasePruner
    if args.pruning:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, args.workers))
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name="ensemble_study",
        direction=args.objective_direction,
        sampler=sampler,
        pruner=pruner,
    )
    # S0 — ensemble winner (direct seed)
    study.enqueue_trial({"theta_max": 0.6445, "angle_penalty": 2.908, "length_penalty": 0.131, "layer01_radial_tolerance": 0.286, "curvature_bonus": 0.571, "curvature_penalty": 0.555, "curvature_tolerance": 0.021})

    # S1 — trial 52 (near-twin of ensemble winner, also 1.0 on its dataset)
    study.enqueue_trial({"theta_max": 0.640, "angle_penalty": 2.869, "length_penalty": 0.131, "layer01_radial_tolerance": 0.285, "curvature_bonus": 0.552, "curvature_penalty": 0.562, "curvature_tolerance": 0.022})

    # S2 — trial 43 regime (lower angle_penalty, balanced curvature)
    study.enqueue_trial({"theta_max": 0.667, "angle_penalty": 2.165, "length_penalty": 0.116, "layer01_radial_tolerance": 0.250, "curvature_bonus": 0.454, "curvature_penalty": 0.454, "curvature_tolerance": 0.036})

    # S3 — trial 45 regime (wider theta, mid-range angle_penalty)
    study.enqueue_trial({"theta_max": 0.755, "angle_penalty": 2.0, "length_penalty": 0.100, "layer01_radial_tolerance": 0.283, "curvature_bonus": 0.462, "curvature_penalty": 0.456, "curvature_tolerance": 0.033})

    # S4 — push curvature_penalty higher (strongest positive correlation r=+0.29)
    study.enqueue_trial({"theta_max": 0.65, "angle_penalty": 2.8, "length_penalty": 0.130, "layer01_radial_tolerance": 0.30, "curvature_bonus": 0.50, "curvature_penalty": 0.75, "curvature_tolerance": 0.020})

    # S5 — push l01_radial_tolerance higher (second strongest signal r=+0.21)
    study.enqueue_trial({"theta_max": 0.65, "angle_penalty": 2.9, "length_penalty": 0.130, "layer01_radial_tolerance": 0.34, "curvature_bonus": 0.55, "curvature_penalty": 0.55, "curvature_tolerance": 0.020})

    # S6 — tighter curvature_tolerance (explore stricter curvature gate)
    study.enqueue_trial({"theta_max": 0.65, "angle_penalty": 3.0, "length_penalty": 0.140, "layer01_radial_tolerance": 0.28, "curvature_bonus": 0.60, "curvature_penalty": 0.60, "curvature_tolerance": 0.012})

    # S7 — lower length_penalty + wider theta (different tradeoff point)
    study.enqueue_trial({"theta_max": 0.73, "angle_penalty": 2.5, "length_penalty": 0.060, "layer01_radial_tolerance": 0.29, "curvature_bonus": 0.40, "curvature_penalty": 0.55, "curvature_tolerance": 0.028})
    study.enqueue_trial({"theta_max": 0.8, "angle_penalty": 2.5, "layer01_radial_penalty": 7.0, "length_penalty": 0.05, "layer01_radial_tolerance": 0.12, "curvature_bonus": 0.7, "curvature_penalty": 0.2, "curvature_tolerance": 0.1})
    
    study.optimize(objective, n_trials=total_trials, n_jobs=args.workers)

    summary_df = pd.DataFrame(all_rows)
    if summary_df.empty:
        print("ERROR: no dataset evaluations were recorded")
        return 1
    if "objective_value" not in summary_df.columns:
        summary_df["objective_value"] = summary_df["objective_metric_value"]

    ascending = args.objective_direction == "minimize"
    summary_csv = sweep_root / "summary_trials.csv"
    summary_df.sort_values(["dataset_id", "objective_value"], ascending=[True, ascending]).to_csv(
        summary_csv, index=False
    )

    for dataset in datasets:
        dataset_id = dataset["dataset_id"]
        dataset_df = summary_df[summary_df["dataset_id"] == dataset_id].copy()
        dataset_csv = sweep_root / f"{dataset_id}_trials.csv"
        dataset_df.sort_values("objective_value", ascending=ascending).to_csv(dataset_csv, index=False)

    completed_df = summary_df[summary_df["state"] == "COMPLETE"].copy()
    if completed_df.empty:
        best_df = pd.DataFrame()
    else:
        best_df = (
            completed_df.sort_values("objective_value", ascending=ascending)
            .groupby("dataset_id", as_index=False)
            .first()
        )
    best_csv = sweep_root / "best_per_dataset.csv"
    best_df.to_csv(best_csv, index=False)

    ensemble_df = pd.DataFrame(ensemble_rows)
    ensemble_csv = sweep_root / "ensemble_trials.csv"
    if not ensemble_df.empty:
        ensemble_df.sort_values("objective_value", ascending=ascending).to_csv(ensemble_csv, index=False)
    else:
        ensemble_df.to_csv(ensemble_csv, index=False)

    if ensemble_df.empty:
        ensemble_best_df = pd.DataFrame()
    else:
        complete_ensemble_df = ensemble_df[ensemble_df["state"] == "COMPLETE"].copy()
        if complete_ensemble_df.empty:
            complete_ensemble_df = ensemble_df.copy()
        ensemble_best_df = complete_ensemble_df.sort_values("objective_value", ascending=ascending).head(1)
    ensemble_best_csv = sweep_root / "ensemble_best.csv"
    ensemble_best_df.to_csv(ensemble_best_csv, index=False)

    with (sweep_root / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "created_at": stamp,
                "datasets": args.datasets,
                "workers": args.workers,
                "ensemble_workers": args.ensemble_workers,
                "trials_per_dataset": args.trials_per_dataset,
                "total_trials": int(total_trials),
                "summary_csv": str(summary_csv),
                "best_csv": str(best_csv),
                "ensemble_trials_csv": str(ensemble_csv),
                "ensemble_best_csv": str(ensemble_best_csv),
                "metrics_tracked": [
                    "segment_precision",
                    "segment_recall",
                    "track_efficiency",
                    "track_fake_rate",
                    "n_bifurcations",
                ],
                "objective": f"{args.objective_direction}_{args.ensemble_statistic}_{args.objective_metric}",
                "objective_metric": args.objective_metric,
                "objective_direction": args.objective_direction,
                "ensemble_statistic": args.ensemble_statistic,
                "ensemble_size": args.datasets,
                "trim_fraction": args.trim_fraction,
                "pruning": args.pruning,
                "min_datasets_before_pruning": args.min_datasets_before_pruning,
                "prune_on_fake_rate": args.max_fake_rate,
                "prune_on_bifurcations": args.max_bifurcations,
            },
            fh,
            indent=2,
        )

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {best_csv}")
    print(f"Wrote: {ensemble_csv}")
    print(f"Wrote: {ensemble_best_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
