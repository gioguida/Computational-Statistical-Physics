#!/usr/bin/env python3

"""
File: scripts/run_dataset_sweep.py
Purpose: Automation scripts and runtime configuration.
Usage: Run from repository root with configured environment.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import json
import math
import subprocess
import sys
import threading
import time
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


def run_cmd(cmd: list[str], cwd: Path, timeout_s: float | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, timeout=timeout_s)


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
    trial_deadline = job.get("trial_deadline")

    def remaining_trial_timeout() -> float | None:
        if trial_deadline is None:
            return None
        remaining = float(trial_deadline) - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError("trial timeout reached before starting command")
        return remaining

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
            str(p["merge_penalty"]),
            "--fork-penalty",
            str(p["fork_penalty"]),
            "--angle-penalty",
            str(p["angle_penalty"]),
        ],
        cwd=project_root,
        timeout_s=remaining_trial_timeout(),
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
            str(p["t_max"]),
            "--n-steps",
            str(int(p["n_steps"])),
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
            str(int(p["eq_sweeps"])),
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
        timeout_s=remaining_trial_timeout(),
    )

    metrics = visualize_metrics(
        {
            "project_root": str(project_root),
            "run_dir": str(run_dir),
            "training_hits_csv": str(hits_csv),
            "ground_truth_csv": str(gt_csv),
            "n_layers": int(job["n_layers"]),
            "merge_penalty": float(p["merge_penalty"]),
            "fork_penalty": float(p["fork_penalty"]),
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


def _resolve_float_bounds(values: list[float] | None, default: float, name: str) -> tuple[float, float]:
    if values is None:
        return (float(default), float(default))
    return _bounds(values, name)


def _resolve_int_bounds(values: list[int] | None, default: int) -> tuple[int, int]:
    if values is None:
        return (int(default), int(default))
    lo = int(min(values))
    hi = int(max(values))
    return (lo, hi)


def _is_swept(lo: float | int, hi: float | int) -> bool:
    return lo != hi


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
    parser.add_argument(
        "--layer01-radial-penalty",
        "--layer-radius-penalty",
        dest="layer01_radial_penalty",
        nargs="+",
        type=float,
        default=[0.0, 15.0],
    )
    parser.add_argument("--length-penalty", nargs="+", type=float, required=True)
    parser.add_argument("--layer01-radial-tolerance", nargs="+", type=float, required=True)
    parser.add_argument("--curvature-bonus", nargs="+", type=float, required=True)
    parser.add_argument("--curvature-penalty", nargs="+", type=float, required=True)
    parser.add_argument("--curvature-tolerance", nargs="+", type=float, required=True)
    parser.add_argument("--t-max", nargs=2, type=float, default=None)
    parser.add_argument("--n-steps", nargs=2, type=int, default=None)
    parser.add_argument("--eq-sweeps", nargs=2, type=int, default=None)
    parser.add_argument("--merge-penalty", nargs=2, type=float, default=None)
    parser.add_argument("--fork-penalty", nargs=2, type=float, default=None)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--max-fake-rate", type=float, default=None)
    parser.add_argument("--max-bifurcations", type=int, default=None)
    parser.add_argument("--objective-metric", default="track_efficiency")
    parser.add_argument("--objective-direction", choices=("maximize", "minimize"), default="maximize")
    parser.add_argument("--ensemble-statistic", choices=ENSEMBLE_STATISTICS, default="mean")
    parser.add_argument("--trim-fraction", type=float, default=0.1)
    parser.add_argument("--pruning", action="store_true")
    parser.add_argument("--min-datasets-before-pruning", type=int, default=3)
    parser.add_argument("--head-starts", default=None)
    parser.add_argument("--trial-timeout", type=float, default=0.0)
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
    if args.trial_timeout < 0.0:
        raise ValueError("--trial-timeout must be non-negative")


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
    t_max_lo, t_max_hi = _resolve_float_bounds(args.t_max, float(annealing_base["t_max"]), "t_max")
    n_steps_lo, n_steps_hi = _resolve_int_bounds(args.n_steps, int(annealing_base["n_steps"]))
    eq_sweeps_lo, eq_sweeps_hi = _resolve_int_bounds(args.eq_sweeps, int(annealing_base["eq_sweeps"]))
    merge_penalty_lo, merge_penalty_hi = _resolve_float_bounds(
        args.merge_penalty, float(inter_cfg.get("merge_penalty", 10.0)), "merge_penalty"
    )
    fork_penalty_lo, fork_penalty_hi = _resolve_float_bounds(
        args.fork_penalty, float(inter_cfg.get("fork_penalty", 10.0)), "fork_penalty"
    )
    if _is_swept(t_max_lo, t_max_hi) and t_max_lo <= 0.0:
        raise ValueError("--t-max lower bound must be > 0 when sweeping (log scale)")
    if n_steps_lo <= 0 or n_steps_hi <= 0:
        raise ValueError("--n-steps bounds must be positive")
    if eq_sweeps_lo <= 0 or eq_sweeps_hi <= 0:
        raise ValueError("--eq-sweeps bounds must be positive")

    param_bounds: dict[str, tuple[float | int, float | int]] = {
        "theta_max": (theta_lo, theta_hi),
        "angle_penalty": (angle_lo, angle_hi),
        "layer01_radial_penalty": (layer_radius_lo, layer_radius_hi),
        "length_penalty": (length_lo, length_hi),
        "layer01_radial_tolerance": (tol_lo, tol_hi),
        "curvature_bonus": (curv_bonus_lo, curv_bonus_hi),
        "curvature_penalty": (curv_penalty_lo, curv_penalty_hi),
        "curvature_tolerance": (curv_tol_lo, curv_tol_hi),
        "t_max": (t_max_lo, t_max_hi),
        "n_steps": (n_steps_lo, n_steps_hi),
        "eq_sweeps": (eq_sweeps_lo, eq_sweeps_hi),
        "merge_penalty": (merge_penalty_lo, merge_penalty_hi),
        "fork_penalty": (fork_penalty_lo, fork_penalty_hi),
    }
    swept_dims = [name for name, (lo, hi) in param_bounds.items() if _is_swept(lo, hi)]
    fixed_dims = {name: lo for name, (lo, hi) in param_bounds.items() if not _is_swept(lo, hi)}
    fixed_dims["t_min"] = float(annealing_base["t_min"])

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
            f"Parallelism: {args.workers} trials x {args.ensemble_workers} ensemble workers "
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
    print(f"Sweep dimensions ({len(swept_dims)}): {', '.join(swept_dims) if swept_dims else '(none)'}", flush=True)
    fixed_summary = ", ".join(f"{key}={value}" for key, value in fixed_dims.items())
    print(f"Fixed parameters: {fixed_summary}", flush=True)

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

    def evaluate_dataset(
        trial_number: int,
        params: dict[str, float | int],
        dataset: dict[str, Any],
        trial_deadline: float | None,
    ) -> dict[str, Any]:
        run_id = f"{dataset['dataset_id']}_trial_{trial_number:04d}"
        run_dir = runs_root / run_id
        base_row: dict[str, Any] = {
            "dataset_id": dataset["dataset_id"],
            "dataset_seed": dataset["dataset_seed"],
            "trial_number": trial_number,
            "run_id": run_id,
            **params,
            "T_max": float(params["t_max"]),
            "T_min": float(annealing_base["t_min"]),
            "n_steps": int(params["n_steps"]),
            "eq_sweeps": int(params["eq_sweeps"]),
            "merge_penalty": float(params["merge_penalty"]),
            "fork_penalty": float(params["fork_penalty"]),
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
                    "annealing_base": annealing_base,
                    "anneal_seed": int(ann_cfg.get("seed", 42)),
                    "params": params,
                    "trial_deadline": trial_deadline,
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
        except subprocess.TimeoutExpired as exc:
            print(
                (
                    f"WARNING: trial {trial_number} dataset {dataset['dataset_id']} timed out: "
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
                "state": "TIMEOUT",
                "error": str(exc),
            }
        except TimeoutError as exc:
            print(
                (
                    f"WARNING: trial {trial_number} dataset {dataset['dataset_id']} timed out: "
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
                "state": "TIMEOUT",
                "error": str(exc),
            }
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
        if _is_swept(theta_lo, theta_hi):
            theta_max = trial.suggest_float("theta_max", theta_lo, theta_hi)
        else:
            theta_max = theta_lo
        if _is_swept(angle_lo, angle_hi):
            angle_penalty = trial.suggest_float("angle_penalty", angle_lo, angle_hi)
        else:
            angle_penalty = angle_lo
        if _is_swept(layer_radius_lo, layer_radius_hi):
            layer01_radial_penalty = trial.suggest_float(
                "layer01_radial_penalty", layer_radius_lo, layer_radius_hi
            )
        else:
            layer01_radial_penalty = layer_radius_lo
        if _is_swept(length_lo, length_hi):
            length_penalty = trial.suggest_float("length_penalty", length_lo, length_hi)
        else:
            length_penalty = length_lo
        if _is_swept(tol_lo, tol_hi):
            layer01_radial_tolerance = trial.suggest_float(
                "layer01_radial_tolerance", tol_lo, tol_hi
            )
        else:
            layer01_radial_tolerance = tol_lo
        if _is_swept(curv_bonus_lo, curv_bonus_hi):
            curvature_bonus = trial.suggest_float("curvature_bonus", curv_bonus_lo, curv_bonus_hi)
        else:
            curvature_bonus = curv_bonus_lo
        if _is_swept(curv_penalty_lo, curv_penalty_hi):
            curvature_penalty = trial.suggest_float(
                "curvature_penalty", curv_penalty_lo, curv_penalty_hi
            )
        else:
            curvature_penalty = curv_penalty_lo
        if _is_swept(curv_tol_lo, curv_tol_hi):
            curvature_tolerance = trial.suggest_float(
                "curvature_tolerance", curv_tol_lo, curv_tol_hi
            )
        else:
            curvature_tolerance = curv_tol_lo
        if _is_swept(t_max_lo, t_max_hi):
            t_max = trial.suggest_float("t_max", float(t_max_lo), float(t_max_hi), log=True)
        else:
            t_max = float(t_max_lo)
        if _is_swept(n_steps_lo, n_steps_hi):
            n_steps = trial.suggest_int("n_steps", int(n_steps_lo), int(n_steps_hi), step=50)
        else:
            n_steps = int(n_steps_lo)
        if _is_swept(eq_sweeps_lo, eq_sweeps_hi):
            eq_sweeps = trial.suggest_int("eq_sweeps", int(eq_sweeps_lo), int(eq_sweeps_hi), step=10)
        else:
            eq_sweeps = int(eq_sweeps_lo)
        if _is_swept(merge_penalty_lo, merge_penalty_hi):
            merge_penalty = trial.suggest_float("merge_penalty", merge_penalty_lo, merge_penalty_hi)
        else:
            merge_penalty = merge_penalty_lo
        if _is_swept(fork_penalty_lo, fork_penalty_hi):
            fork_penalty = trial.suggest_float("fork_penalty", fork_penalty_lo, fork_penalty_hi)
        else:
            fork_penalty = fork_penalty_lo

        params = {
            "theta_max": float(theta_max),
            "angle_penalty": float(angle_penalty),
            "layer01_radial_penalty": float(layer01_radial_penalty),
            "length_penalty": float(length_penalty),
            "layer01_radial_tolerance": float(layer01_radial_tolerance),
            "curvature_bonus": float(curvature_bonus),
            "curvature_penalty": float(curvature_penalty),
            "curvature_tolerance": float(curvature_tolerance),
            "t_max": float(t_max),
            "t_min": float(annealing_base["t_min"]),
            "n_steps": int(n_steps),
            "eq_sweeps": int(eq_sweeps),
            "merge_penalty": float(merge_penalty),
            "fork_penalty": float(fork_penalty),
        }

        values_by_dataset: dict[str, float] = {}
        dataset_rows: list[dict[str, Any]] = []
        state = "COMPLETE"
        prune_reason = None
        trial_deadline = time.monotonic() + args.trial_timeout if args.trial_timeout > 0 else None

        with futures.ThreadPoolExecutor(max_workers=args.ensemble_workers) as executor:
            submitted = {
                executor.submit(evaluate_dataset, trial.number, params, dataset, trial_deadline): dataset
                for dataset in datasets
            }
            completed_count = 0
            try:
                as_completed_timeout = args.trial_timeout if args.trial_timeout > 0 else None
                for future in futures.as_completed(submitted, timeout=as_completed_timeout):
                    completed_count += 1
                    row = future.result()
                    dataset_rows.append(row)
                    values_by_dataset[row["dataset_id"]] = float(row["objective_metric_value"])
                    if row.get("state") == "TIMEOUT":
                        state = "TIMEOUT"
                        prune_reason = f"trial timed out after {args.trial_timeout:.2f}s"
                        for pending in submitted:
                            pending.cancel()
                        break

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
            except futures.TimeoutError:
                state = "TIMEOUT"
                prune_reason = f"trial timed out after {args.trial_timeout:.2f}s"
                print(
                    f"WARNING: trial {trial.number} exceeded timeout of {args.trial_timeout:.2f}s",
                    file=sys.stderr,
                    flush=True,
                )
                for pending in submitted:
                    pending.cancel()

        values = [values_by_dataset.get(dataset["dataset_id"], float("nan")) for dataset in datasets]
        if state == "TIMEOUT":
            objective_value = 0.0 if args.objective_direction == "maximize" else float("inf")
        else:
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
            "T_max": float(params["t_max"]),
            "T_min": float(params["t_min"]),
            "n_steps": int(params["n_steps"]),
            "eq_sweeps": int(params["eq_sweeps"]),
            "merge_penalty": float(params["merge_penalty"]),
            "fork_penalty": float(params["fork_penalty"]),
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
    if args.head_starts is not None:
        head_starts_path = Path(args.head_starts).expanduser()
        if not head_starts_path.is_absolute():
            head_starts_path = (PROJECT_ROOT / head_starts_path).resolve()
        with head_starts_path.open("r", encoding="utf-8") as fh:
            head_starts_data = json.load(fh)
        if not isinstance(head_starts_data, list):
            raise ValueError("--head-starts JSON must contain a list of parameter mappings")

        swept_keys = set(swept_dims)
        enqueued = 0
        for item in head_starts_data:
            if not isinstance(item, dict):
                continue
            candidate: dict[str, Any] = {}
            for key, value in item.items():
                if key not in swept_keys:
                    continue
                if key in {"n_steps", "eq_sweeps"}:
                    candidate[key] = int(value)
                else:
                    candidate[key] = float(value)
            if candidate:
                study.enqueue_trial(candidate)
                enqueued += 1
        print(f"Enqueued {enqueued} head-start trials from {head_starts_path}", flush=True)

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
                "trial_timeout_seconds": args.trial_timeout,
                "head_starts": args.head_starts,
                "sweep_dimensions": swept_dims,
                "fixed_parameters": fixed_dims,
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

