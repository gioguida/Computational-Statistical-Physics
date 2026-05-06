#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
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


SEARCH_PARAM_KEYS = (
    "theta_max",
    "angle_penalty",
    "layer_radius_penalty",
    "length_penalty",
    "layer01_radial_tolerance",
    "anneal_t_max",
)

RANKING_COLUMNS = [
    "search_score",
    "track_efficiency",
    "track_efficiency_soft",
    "segment_precision",
    "segment_recall",
    "track_fake_rate",
    "n_bifurcations",
]
RANKING_ASCENDING = [False, False, False, False, False, True, True]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be mapping: {path}")
    return cfg


def generate_dataset(cfg: DataConfig, seed: int, out_dir: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)

    # Detector still uses the legacy global NumPy RNG, so pin it here to make
    # the sweep reproducible dataset-by-dataset.
    legacy_state = np.random.get_state()
    try:
        np.random.seed(seed)
        experiment = Detector(
            cfg.detector_layers,
            cfg.n_particles,
            cfg.traj_radius_low,
            cfg.traj_radius_high,
        )
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


def _compute_search_score(row: dict[str, Any]) -> float:
    # Keep strict track efficiency dominant, but add softer terms so Optuna can
    # distinguish between near-tied trials instead of seeing a mostly discrete objective.
    return (
        1000.0 * float(row.get("track_efficiency", 0.0))
        + 50.0 * float(row.get("track_efficiency_soft", 0.0))
        + 10.0 * float(row.get("segment_precision", 0.0))
        + 5.0 * float(row.get("segment_recall", 0.0))
        - 10.0 * float(row.get("track_fake_rate", 0.0))
        - 5.0 * float(row.get("n_bifurcations", 0))
    )


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
            "--hits-csv", str(hits_csv),
            "--out-dir", str(inter_dir),
            "--theta-max", str(p["theta_max"]),
            "--merge-penalty", str(job["merge_penalty"]),
            "--fork-penalty", str(job["fork_penalty"]),
            "--angle-penalty", str(p["angle_penalty"]),
        ],
        cwd=project_root,
    )

    run_cmd(
        [
            str(annealing_bin),
            "--hits-csv", str(hits_csv),
            "--segments-csv", str(inter_dir / "segments.csv"),
            "--edges-csv", str(inter_dir / "J_edges.csv"),
            "--out-dir", str(ann_dir),
            "--t-min", str(ann["t_min"]),
            "--t-max", str(p["anneal_t_max"]),
            "--n-steps", str(ann["n_steps"]),
            "--toll", str(ann["toll"]),
            "--length-penalty", str(p["length_penalty"]),
            "--layer-radius-penalty", str(p["layer_radius_penalty"]),
            "--layer01-radial-tolerance", str(p["layer01_radial_tolerance"]),
            "--first-gap", str(job["first_gap"]),
            "--eq-sweeps", str(ann["eq_sweeps"]),
            "--log-every-steps", str(ann["log_every_steps"]),
            "--checkpoint-every-steps", str(ann["checkpoint_every_steps"]),
            "--seed", str(job["anneal_seed"]),
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
            "create_plots": bool(job["create_plots"]),
            "verbose": False,
        }
    )

    row = {
        "run_id": run_dir.name,
        "merge_penalty": float(job["merge_penalty"]),
        "fork_penalty": float(job["fork_penalty"]),
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
    row["search_score"] = _compute_search_score(row)
    return row


def _bounds(values: list[float], name: str) -> tuple[float, float]:
    if not values:
        raise ValueError(f"Missing values for {name}")
    lo = min(values)
    hi = max(values)
    return (float(lo), float(hi))


def _sort_trials(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(RANKING_COLUMNS, ascending=RANKING_ASCENDING)


def _clip_seed_trial(
    seed_trial: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> dict[str, float]:
    clipped: dict[str, float] = {}
    for key, value in seed_trial.items():
        lo, hi = bounds[key]
        clipped[key] = float(min(max(value, lo), hi))
    return clipped


def _native_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _row_to_native_dict(row: pd.Series) -> dict[str, Any]:
    return {key: _native_value(value) for key, value in row.items()}


def _series_stats(series: pd.Series) -> dict[str, float]:
    clean = series.astype(float)
    return {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset-aware Bayesian optimization with Optuna")
    parser.add_argument("--config", default="scripts/config.yaml")
    parser.add_argument("--datasets", type=int, default=8)
    parser.add_argument("--trials-per-dataset", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seeds-start", type=int, default=1000)
    parser.add_argument("--output-root", default="results/sweeps")
    parser.add_argument("--theta-max", nargs="+", type=float, required=True)
    parser.add_argument("--angle-penalty", nargs="+", type=float, required=True)
    parser.add_argument("--layer-radius-penalty", nargs="+", type=float, required=True)
    parser.add_argument("--length-penalty", nargs="+", type=float, required=True)
    parser.add_argument("--layer01-radial-tolerance", nargs="+", type=float, required=True)
    parser.add_argument("--anneal-t-max", nargs="+", type=float, required=True)
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--max-fake-rate", type=float, default=None)
    parser.add_argument("--max-bifurcations", type=int, default=None)
    parser.add_argument("--log-every-steps", type=int, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--create-plots", action="store_true")
    parser.add_argument("--leaderboard-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    theta_lo, theta_hi = _bounds(args.theta_max, "theta_max")
    angle_lo, angle_hi = _bounds(args.angle_penalty, "angle_penalty")
    layer_radius_lo, layer_radius_hi = _bounds(args.layer_radius_penalty, "layer_radius_penalty")
    length_lo, length_hi = _bounds(args.length_penalty, "length_penalty")
    tol_lo, tol_hi = _bounds(args.layer01_radial_tolerance, "layer01_radial_tolerance")
    anneal_tmax_lo, anneal_tmax_hi = _bounds(args.anneal_t_max, "anneal_t_max")

    annealing_base = {
        "t_min": float(ann_cfg.get("t_min", 1e-3)),
        "n_steps": int(ann_cfg.get("n_steps", 300)),
        "toll": float(ann_cfg.get("toll", 1e-6)),
        "eq_sweeps": int(ann_cfg.get("eq_sweeps", 100)),
        "log_every_steps": int(args.log_every_steps if args.log_every_steps is not None else ann_cfg.get("log_every_steps", 10)),
        "checkpoint_every_steps": int(
            args.checkpoint_every_steps
            if args.checkpoint_every_steps is not None
            else ann_cfg.get("checkpoint_every_steps", 10)
        ),
    }

    search_bounds = {
        "theta_max": (theta_lo, theta_hi),
        "angle_penalty": (angle_lo, angle_hi),
        "layer_radius_penalty": (layer_radius_lo, layer_radius_hi),
        "length_penalty": (length_lo, length_hi),
        "layer01_radial_tolerance": (tol_lo, tol_hi),
        "anneal_t_max": (anneal_tmax_lo, anneal_tmax_hi),
    }

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = (PROJECT_ROOT / args.output_root / stamp).resolve()
    datasets_root = sweep_root / "datasets"
    runs_root = sweep_root / "runs"
    datasets_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    print(f"Sweep root: {sweep_root}")
    print(f"Datasets: {args.datasets}")
    print(f"Trials per dataset: {args.trials_per_dataset}")
    print(f"Workers: {args.workers}")
    print(f"Create plots: {bool(args.create_plots)}")

    all_rows: list[dict[str, Any]] = []

    seed_trials = [
        {
            "theta_max": 0.33,
            "angle_penalty": 6.0,
            "layer_radius_penalty": 6.2,
            "length_penalty": 0.52,
            "layer01_radial_tolerance": 0.16,
            "anneal_t_max": 4.0,
        },
        {
            "theta_max": 0.35,
            "angle_penalty": 5.5,
            "layer_radius_penalty": 5.8,
            "length_penalty": 0.42,
            "layer01_radial_tolerance": 0.17,
            "anneal_t_max": 5.0,
        },
        {
            "theta_max": 0.31,
            "angle_penalty": 7.0,
            "layer_radius_penalty": 7.0,
            "length_penalty": 0.45,
            "layer01_radial_tolerance": 0.14,
            "anneal_t_max": 5.0,
        },
        {
            "theta_max": 0.32,
            "angle_penalty": 6.5,
            "layer_radius_penalty": 5.0,
            "length_penalty": 0.38,
            "layer01_radial_tolerance": 0.20,
            "anneal_t_max": 6.0,
        },
        {
            "theta_max": 0.37,
            "angle_penalty": 5.0,
            "layer_radius_penalty": 6.0,
            "length_penalty": 0.30,
            "layer01_radial_tolerance": 0.18,
            "anneal_t_max": 6.5,
        },
    ]

    for d in range(args.datasets):
        dataset_id = f"ds_{d:03d}"
        dataset_dir = datasets_root / dataset_id
        dataset_seed = args.seeds_start + d
        train_csv, gt_csv = generate_dataset(gen_cfg, dataset_seed, dataset_dir)

        dataset_rows: list[dict[str, Any]] = []

        def objective(trial: optuna.Trial) -> float:
            theta_max = trial.suggest_float("theta_max", theta_lo, theta_hi)
            angle_penalty = trial.suggest_float("angle_penalty", angle_lo, angle_hi)
            layer_radius_penalty = trial.suggest_float("layer_radius_penalty", layer_radius_lo, layer_radius_hi)
            length_penalty = trial.suggest_float("length_penalty", length_lo, length_hi)
            layer01_radial_tolerance = trial.suggest_float("layer01_radial_tolerance", tol_lo, tol_hi)
            anneal_t_max = trial.suggest_float("anneal_t_max", anneal_tmax_lo, anneal_tmax_hi)

            run_id = f"{dataset_id}_trial_{trial.number:04d}"
            run_dir = runs_root / run_id

            row = run_one_job(
                {
                    "project_root": str(PROJECT_ROOT),
                    "build_dir": str(build_dir),
                    "run_dir": str(run_dir),
                    "hits_csv": str(train_csv),
                    "gt_csv": str(gt_csv),
                    "n_layers": len(detector_layers),
                    "first_gap": first_gap,
                    "merge_penalty": merge_penalty,
                    "fork_penalty": fork_penalty,
                    "annealing_base": annealing_base,
                    "anneal_seed": int(ann_cfg.get("seed", 42)),
                    "create_plots": bool(args.create_plots),
                    "params": {
                        "theta_max": theta_max,
                        "angle_penalty": angle_penalty,
                        "layer_radius_penalty": layer_radius_penalty,
                        "length_penalty": length_penalty,
                        "layer01_radial_tolerance": layer01_radial_tolerance,
                        "anneal_t_max": anneal_t_max,
                    },
                }
            )
            row["dataset_id"] = dataset_id
            row["trial_number"] = trial.number

            trial.set_user_attr("search_score", row["search_score"])
            trial.set_user_attr("segment_precision", row["segment_precision"])
            trial.set_user_attr("segment_recall", row["segment_recall"])
            trial.set_user_attr("track_efficiency", row["track_efficiency"])
            trial.set_user_attr("track_efficiency_soft", row.get("track_efficiency_soft", 0.0))
            trial.set_user_attr("track_fake_rate", row["track_fake_rate"])
            trial.set_user_attr("n_bifurcations", row["n_bifurcations"])
            trial.set_user_attr("run_id", row["run_id"])

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

            row["pruned"] = pruned
            row["prune_reason"] = prune_reason
            dataset_rows.append(row)

            if pruned:
                raise optuna.TrialPruned(prune_reason)

            return float(row["search_score"])

        sampler = optuna.samplers.TPESampler(seed=args.sampler_seed + d)
        study = optuna.create_study(
            study_name=f"{dataset_id}_study",
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=max(8, args.workers)),
        )
        for seed_trial in seed_trials:
            study.enqueue_trial(_clip_seed_trial(seed_trial, search_bounds))

        study.optimize(objective, n_trials=args.trials_per_dataset, n_jobs=args.workers)

        dataset_df = pd.DataFrame(dataset_rows)
        if dataset_df.empty or "track_efficiency" not in dataset_df.columns:
            print(f"WARNING: no successful trials for {dataset_id}, skipping")
            continue
        dataset_df["objective_value"] = dataset_df["search_score"]

        dataset_csv = sweep_root / f"{dataset_id}_trials.csv"
        _sort_trials(dataset_df).to_csv(dataset_csv, index=False)
        all_rows.extend(dataset_rows)

    summary_df = pd.DataFrame(all_rows)
    if summary_df.empty or "track_efficiency" not in summary_df.columns:
        print("ERROR: no successful trials across all datasets")
        return 1
    summary_df["objective_value"] = summary_df["search_score"]

    summary_csv = sweep_root / "summary_trials.csv"
    _sort_trials(summary_df).to_csv(summary_csv, index=False)

    completed_df = summary_df[~summary_df["pruned"].astype(bool)].copy()
    if completed_df.empty:
        print("ERROR: all trials were pruned")
        return 1

    leaderboard_df = _sort_trials(completed_df).reset_index(drop=True)
    leaderboard_csv = sweep_root / "leaderboard_trials.csv"
    leaderboard_df.head(args.leaderboard_size).to_csv(leaderboard_csv, index=False)

    best_df = (
        _sort_trials(completed_df)
        .groupby("dataset_id", as_index=False)
        .first()
    )
    best_csv = sweep_root / "best_per_dataset.csv"
    best_df.to_csv(best_csv, index=False)

    best_trial = leaderboard_df.iloc[0]
    best_trial_json = sweep_root / "best_single_trial.json"
    with best_trial_json.open("w", encoding="utf-8") as fh:
        json.dump(_row_to_native_dict(best_trial), fh, indent=2)
        fh.write("\n")

    recommended_params = {
        key: float(best_df[key].astype(float).median())
        for key in SEARCH_PARAM_KEYS
    }
    recommended_payload = {
        "selection_rule": "median_of_best_per_dataset_sorted_by_search_score",
        "search_score_formula": (
            "1000*track_efficiency + 50*track_efficiency_soft + "
            "10*segment_precision + 5*segment_recall - "
            "10*track_fake_rate - 5*n_bifurcations"
        ),
        "n_dataset_bests": int(len(best_df)),
        "recommended_params": recommended_params,
        "parameter_stats_over_dataset_bests": {
            key: _series_stats(best_df[key])
            for key in SEARCH_PARAM_KEYS
        },
        "fixed_params": {
            "merge_penalty": merge_penalty,
            "fork_penalty": fork_penalty,
            "anneal_t_min": annealing_base["t_min"],
            "anneal_n_steps": annealing_base["n_steps"],
            "anneal_eq_sweeps": annealing_base["eq_sweeps"],
        },
        "mean_metrics_over_dataset_bests": {
            "search_score": float(best_df["search_score"].mean()),
            "track_efficiency": float(best_df["track_efficiency"].mean()),
            "track_efficiency_soft": float(best_df["track_efficiency_soft"].mean()),
            "segment_precision": float(best_df["segment_precision"].mean()),
            "segment_recall": float(best_df["segment_recall"].mean()),
            "track_fake_rate": float(best_df["track_fake_rate"].mean()),
            "n_bifurcations": float(best_df["n_bifurcations"].mean()),
        },
        "best_single_trial": _row_to_native_dict(best_trial),
    }

    recommended_json = sweep_root / "recommended_params.json"
    with recommended_json.open("w", encoding="utf-8") as fh:
        json.dump(recommended_payload, fh, indent=2)
        fh.write("\n")

    recommended_config = {
        "interaction": {
            "theta_max": recommended_params["theta_max"],
            "merge_penalty": merge_penalty,
            "fork_penalty": fork_penalty,
            "angle_penalty": recommended_params["angle_penalty"],
        },
        "annealing": {
            "t_min": annealing_base["t_min"],
            "t_max": recommended_params["anneal_t_max"],
            "n_steps": annealing_base["n_steps"],
            "length_penalty": recommended_params["length_penalty"],
            "layer_radius_penalty": recommended_params["layer_radius_penalty"],
            "layer01_radial_tolerance": recommended_params["layer01_radial_tolerance"],
            "eq_sweeps": annealing_base["eq_sweeps"],
            "log_every_steps": annealing_base["log_every_steps"],
            "checkpoint_every_steps": annealing_base["checkpoint_every_steps"],
            "seed": int(ann_cfg.get("seed", 42)),
        },
    }
    recommended_yaml = sweep_root / "recommended_config.yaml"
    with recommended_yaml.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(recommended_config, fh, sort_keys=False)

    with (sweep_root / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "created_at": stamp,
                "datasets": args.datasets,
                "workers": args.workers,
                "trials_per_dataset": args.trials_per_dataset,
                "total_trials": int(args.datasets * args.trials_per_dataset),
                "summary_csv": str(summary_csv),
                "leaderboard_csv": str(leaderboard_csv),
                "best_csv": str(best_csv),
                "best_single_trial_json": str(best_trial_json),
                "recommended_params_json": str(recommended_json),
                "recommended_config_yaml": str(recommended_yaml),
                "search_params": list(SEARCH_PARAM_KEYS),
                "search_bounds": {
                    key: {"low": bounds[0], "high": bounds[1]}
                    for key, bounds in search_bounds.items()
                },
                "metrics_tracked": [
                    "segment_precision",
                    "segment_recall",
                    "track_efficiency",
                    "track_efficiency_soft",
                    "track_fake_rate",
                    "n_bifurcations",
                    "search_score",
                ],
                "objective": "maximize_search_score",
                "prune_on_fake_rate": args.max_fake_rate,
                "prune_on_bifurcations": args.max_bifurcations,
                "create_plots": bool(args.create_plots),
            },
            fh,
            indent=2,
        )

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {leaderboard_csv}")
    print(f"Wrote: {best_csv}")
    print(f"Wrote: {best_trial_json}")
    print(f"Wrote: {recommended_json}")
    print(f"Wrote: {recommended_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
