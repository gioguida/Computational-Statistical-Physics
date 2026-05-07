#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import json
import math
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hits.data_gen import DataConfig
from src.hits.hits import Detector
from src.plotting.metrics import visualize_metrics

SUMMARY_METRICS = (
    "search_score",
    "track_efficiency",
    "track_efficiency_soft",
    "precision",
    "TPR",
    "F1",
    "fake_rate",
    "n_bifurcations",
    "freeze_temperature",
    "wall_seconds",
    "n_hits",
    "n_segments",
    "n_nonzero_edges",
    "cpp_worker_peak_estimate_mb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the current fixed parameters in scripts/config.yaml on many "
            "independently generated datasets and aggregate the statistics."
        )
    )
    parser.add_argument("--config", default="scripts/config.yaml")
    parser.add_argument("--datasets", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dataset-seed-start", type=int, default=1000)
    parser.add_argument("--anneal-seed-start", type=int, default=None)
    parser.add_argument("--vary-anneal-seed", action="store_true")
    parser.add_argument("--output-root", default="results/fixed_config_eval")
    parser.add_argument("--scratch-root", default=None)
    parser.add_argument("--keep-run-artifacts", action="store_true")
    parser.add_argument("--create-plots", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return cfg


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> None:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        if result.stdout:
            fh.write(result.stdout)
            if not result.stdout.endswith("\n"):
                fh.write("\n")
        if result.stderr:
            fh.write("\n[stderr]\n")
            fh.write(result.stderr)
            if not result.stderr.endswith("\n"):
                fh.write("\n")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)} "
            f"(see {log_path})"
        )


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def dir_size_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def generate_dataset(cfg: DataConfig, seed: int, out_dir: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)

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
    fake_hits.insert(
        0,
        "hit_id",
        np.arange(n_real_hits, n_real_hits + len(fake_hits), dtype=int),
    )
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


def compute_search_score(row: dict[str, Any]) -> float:
    return (
        1000.0 * float(row.get("track_efficiency", 0.0))
        + 50.0 * float(row.get("track_efficiency_soft", 0.0))
        + 10.0 * float(row.get("precision", 0.0))
        + 5.0 * float(row.get("TPR", 0.0))
        - 10.0 * float(row.get("fake_rate", 0.0))
        - 5.0 * float(row.get("n_bifurcations", 0))
    )


def stats_summary(series: pd.Series) -> dict[str, float]:
    values = series.astype(float).to_numpy()
    n = len(values)
    mean = float(values.mean())
    median = float(np.median(values))
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / math.sqrt(n)) if n > 1 else 0.0
    ci95 = 1.96 * sem
    return {
        "count": float(n),
        "mean": mean,
        "median": median,
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
        "sem": sem,
        "ci95_low": mean - ci95,
        "ci95_high": mean + ci95,
    }


def build_fixed_params(
    cfg: dict[str, Any],
    gen_cfg: DataConfig,
    anneal_seed_start: int | None,
) -> dict[str, Any]:
    inter_cfg = cfg.get("interaction", {})
    ann_cfg = cfg.get("annealing", {})
    if len(gen_cfg.detector_layers) < 2:
        raise ValueError("Need at least two detector layers")

    first_gap = abs(float(gen_cfg.detector_layers[1]) - float(gen_cfg.detector_layers[0]))
    base_seed = int(ann_cfg.get("seed", 42)) if anneal_seed_start is None else int(anneal_seed_start)

    return {
        "theta_max": float(inter_cfg.get("theta_max", 0.0)),
        "merge_penalty": float(inter_cfg.get("merge_penalty", 10.0)),
        "fork_penalty": float(inter_cfg.get("fork_penalty", 10.0)),
        "angle_penalty": float(inter_cfg.get("angle_penalty", 0.0)),
        "t_min": float(ann_cfg.get("t_min", 1e-3)),
        "t_max": float(ann_cfg.get("t_max", 5.0)),
        "n_steps": int(ann_cfg.get("n_steps", 300)),
        "toll": float(ann_cfg.get("toll", 1e-6)),
        "length_penalty": float(ann_cfg.get("length_penalty", 0.0)),
        "layer_radius_penalty": float(ann_cfg.get("layer_radius_penalty", inter_cfg.get("layer_radius_penalty", 0.0))),
        "layer01_radial_tolerance": float(ann_cfg.get("layer01_radial_tolerance", 0.0)),
        "eq_sweeps": int(ann_cfg.get("eq_sweeps", 100)),
        "log_every_steps": int(ann_cfg.get("log_every_steps", 100)),
        "checkpoint_every_steps": int(ann_cfg.get("checkpoint_every_steps", 10)),
        "anneal_seed_base": base_seed,
        "first_gap": first_gap,
    }


def estimate_cpp_worker_peak_mb(n_segments: int, n_edges: int, n_checkpoints: int) -> float:
    directed_edges = 2 * int(n_edges)
    segments_bytes = 48 * int(n_segments)
    adjacency_bytes = 24 * int(n_segments) + 16 * directed_edges
    field_bytes = 8 * int(n_segments)
    state_bytes = 4 * int(n_segments) * (4 + max(int(n_checkpoints), 0))

    # The binaries keep multiple copies of the sparse graph and state over the
    # pipeline. Use a safety multiplier plus a fixed overhead to turn the raw
    # container-size estimate into a scheduler-facing request estimate.
    estimated_bytes = int(
        3.0 * (segments_bytes + adjacency_bytes + field_bytes + state_bytes)
        + 32 * 1024 * 1024
    )
    return estimated_bytes / (1024.0 * 1024.0)


def recommend_total_mem_gb(max_worker_mb: float, workers: int) -> int:
    shared_python_mb = 1024.0
    per_active_job_overhead_mb = 16.0
    raw_total_mb = 1.5 * (shared_python_mb + workers * (max_worker_mb + per_active_job_overhead_mb))
    rounded_gb = int(math.ceil(raw_total_mb / 1024.0))
    if workers >= 32:
        rounded_gb = max(rounded_gb, 8)
    return max(rounded_gb, 4)


def expected_problem_size(gen_cfg: DataConfig) -> dict[str, float]:
    n_layers = len(gen_cfg.detector_layers)
    mean_hits_per_layer = float(gen_cfg.n_particles + gen_cfg.mean_fakes_per_layer)
    expected_total_hits = float(n_layers * mean_hits_per_layer)
    expected_segments = float((n_layers - 1) * (mean_hits_per_layer ** 2))
    return {
        "n_layers": float(n_layers),
        "mean_hits_per_layer": mean_hits_per_layer,
        "expected_total_hits": expected_total_hits,
        "expected_segments": expected_segments,
    }


def native_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def series_to_native_summary(df: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, dict[str, float]]:
    return {
        column: stats_summary(df[column])
        for column in columns
        if column in df.columns
    }


def write_summary_report(
    out_path: Path,
    dataset_rows: pd.DataFrame,
    aggregate_stats: dict[str, dict[str, float]],
    summary_payload: dict[str, Any],
) -> None:
    best_row = dataset_rows.sort_values("search_score", ascending=False).iloc[0]
    worst_row = dataset_rows.sort_values("search_score", ascending=True).iloc[0]

    lines = [
        "# Fixed-Config Evaluation Summary",
        "",
        f"- Datasets: {int(summary_payload['datasets'])}",
        f"- Workers: {int(summary_payload['workers'])}",
        f"- Dataset seed start: {int(summary_payload['dataset_seed_start'])}",
        f"- Vary annealing seed: {bool(summary_payload['vary_anneal_seed'])}",
        f"- Keep run artifacts: {bool(summary_payload['keep_run_artifacts'])}",
        "",
        "## Fixed Parameters",
        "",
    ]

    for key, value in summary_payload["fixed_params"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
        ]
    )
    for key in (
        "track_efficiency",
        "track_efficiency_soft",
        "precision",
        "TPR",
        "F1",
        "fake_rate",
        "n_bifurcations",
        "search_score",
    ):
        if key not in aggregate_stats:
            continue
        stats = aggregate_stats[key]
        lines.append(
            (
                f"- {key}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
                f"95% CI=[{stats['ci95_low']:.6f}, {stats['ci95_high']:.6f}]"
            )
        )

    mem = summary_payload["memory_estimate"]
    lines.extend(
        [
            "",
            "## Problem Size And Memory",
            "",
            f"- max n_hits: {int(dataset_rows['n_hits'].max())}",
            f"- max n_segments: {int(dataset_rows['n_segments'].max())}",
            f"- max n_nonzero_edges: {int(dataset_rows['n_nonzero_edges'].max())}",
            f"- estimated peak C++ memory per active worker: {mem['max_cpp_worker_peak_estimate_mb']:.2f} MB",
            f"- recommended Euler memory request for this worker count: {int(mem['recommended_total_mem_gb'])} GB",
            "",
            "## Extremes",
            "",
            f"- best dataset by search_score: {best_row['dataset_id']} ({best_row['search_score']:.6f})",
            f"- worst dataset by search_score: {worst_row['dataset_id']} ({worst_row['search_score']:.6f})",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_worker_outputs(dataset_dir: Path, run_dir: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_dir / "training_hits.csv", dest_dir / "training_hits.csv")
    shutil.copy2(dataset_dir / "ground_truth_hits.csv", dest_dir / "ground_truth_hits.csv")
    shutil.copytree(run_dir / "interaction", dest_dir / "interaction")
    shutil.copytree(run_dir / "annealing", dest_dir / "annealing")
    if (run_dir / "logs").exists():
        shutil.copytree(run_dir / "logs", dest_dir / "logs")


def main() -> int:
    args = parse_args()

    if args.datasets <= 0:
        raise ValueError("--datasets must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.create_plots and args.workers > 1:
        raise ValueError("--create-plots is only supported with --workers 1")

    config_path = (PROJECT_ROOT / args.config).resolve()
    cfg = load_yaml(config_path)
    gen_cfg = DataConfig.from_yaml(config_path)
    fixed_params = build_fixed_params(cfg, gen_cfg, args.anneal_seed_start)

    build_dir = (PROJECT_ROOT / str(cfg.get("build", {}).get("build_dir", "build"))).resolve()
    interaction_bin = build_dir / "run_interaction"
    annealing_bin = build_dir / "run_annealing"
    if not interaction_bin.exists() or not annealing_bin.exists():
        raise FileNotFoundError("Missing binaries in build dir. Build the project first.")

    workers = min(int(args.workers), int(args.datasets))
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = (PROJECT_ROOT / args.output_root / stamp).resolve()
    scratch_root = (
        Path(args.scratch_root).resolve()
        if args.scratch_root is not None
        else (sweep_root / "_scratch").resolve()
    )
    runs_root = sweep_root / "dataset_runs"
    failures_root = sweep_root / "failures"

    sweep_root.mkdir(parents=True, exist_ok=True)
    if args.keep_run_artifacts:
        runs_root.mkdir(parents=True, exist_ok=True)
    failures_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, sweep_root / "config_snapshot.yaml")

    dataset_slots: queue.Queue[int] = queue.Queue()
    for slot in range(workers):
        dataset_slots.put(slot)

    print_lock = threading.Lock()

    def log(message: str) -> None:
        with print_lock:
            print(message, flush=True)

    def run_dataset(dataset_idx: int) -> dict[str, Any]:
        slot = dataset_slots.get()
        dataset_id = f"ds_{dataset_idx:04d}"
        worker_root = scratch_root / f"worker_{slot:02d}"
        dataset_dir = worker_root / "data"
        run_dir = worker_root / "run"
        try:
            reset_dir(dataset_dir)
            reset_dir(run_dir)

            dataset_seed = int(args.dataset_seed_start) + dataset_idx
            anneal_seed = (
                fixed_params["anneal_seed_base"] + dataset_idx
                if args.vary_anneal_seed
                else fixed_params["anneal_seed_base"]
            )

            log(f"[{dataset_id}] slot={slot} dataset_seed={dataset_seed} anneal_seed={anneal_seed}")
            train_csv, gt_csv = generate_dataset(gen_cfg, dataset_seed, dataset_dir)

            interaction_dir = run_dir / "interaction"
            annealing_dir = run_dir / "annealing"
            interaction_dir.mkdir(parents=True, exist_ok=True)
            annealing_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.perf_counter()
            run_cmd(
                [
                    str(interaction_bin),
                    "--hits-csv",
                    str(train_csv),
                    "--out-dir",
                    str(interaction_dir),
                    "--theta-max",
                    str(fixed_params["theta_max"]),
                    "--merge-penalty",
                    str(fixed_params["merge_penalty"]),
                    "--fork-penalty",
                    str(fixed_params["fork_penalty"]),
                    "--angle-penalty",
                    str(fixed_params["angle_penalty"]),
                ],
                cwd=PROJECT_ROOT,
                log_path=run_dir / "logs" / "interaction.log",
            )
            t1 = time.perf_counter()

            run_cmd(
                [
                    str(annealing_bin),
                    "--hits-csv",
                    str(train_csv),
                    "--segments-csv",
                    str(interaction_dir / "segments.csv"),
                    "--edges-csv",
                    str(interaction_dir / "J_edges.csv"),
                    "--out-dir",
                    str(annealing_dir),
                    "--t-min",
                    str(fixed_params["t_min"]),
                    "--t-max",
                    str(fixed_params["t_max"]),
                    "--n-steps",
                    str(fixed_params["n_steps"]),
                    "--toll",
                    str(fixed_params["toll"]),
                    "--length-penalty",
                    str(fixed_params["length_penalty"]),
                    "--layer-radius-penalty",
                    str(fixed_params["layer_radius_penalty"]),
                    "--layer01-radial-tolerance",
                    str(fixed_params["layer01_radial_tolerance"]),
                    "--first-gap",
                    str(fixed_params["first_gap"]),
                    "--eq-sweeps",
                    str(fixed_params["eq_sweeps"]),
                    "--log-every-steps",
                    str(fixed_params["log_every_steps"]),
                    "--checkpoint-every-steps",
                    str(fixed_params["checkpoint_every_steps"]),
                    "--seed",
                    str(anneal_seed),
                ],
                cwd=PROJECT_ROOT,
                log_path=run_dir / "logs" / "annealing.log",
            )
            t2 = time.perf_counter()

            metrics = visualize_metrics(
                {
                    "project_root": str(PROJECT_ROOT),
                    "run_dir": str(run_dir),
                    "training_hits_csv": str(train_csv),
                    "ground_truth_csv": str(gt_csv),
                    "n_layers": len(gen_cfg.detector_layers),
                    "merge_penalty": fixed_params["merge_penalty"],
                    "fork_penalty": fixed_params["fork_penalty"],
                    "angle_penalty": fixed_params["angle_penalty"],
                    "create_plots": bool(args.create_plots),
                    "verbose": False,
                }
            )
            t3 = time.perf_counter()

            interaction_meta = json.loads((interaction_dir / "interaction_meta.json").read_text(encoding="utf-8"))
            annealing_meta = json.loads((annealing_dir / "annealing_meta.json").read_text(encoding="utf-8"))

            row: dict[str, Any] = {
                "dataset_id": dataset_id,
                "dataset_seed": dataset_seed,
                "anneal_seed": anneal_seed,
                "wall_seconds": t3 - t0,
                "interaction_seconds": t1 - t0,
                "annealing_seconds": t2 - t1,
                "metrics_seconds": t3 - t2,
                "n_hits": int(interaction_meta["n_hits"]),
                "n_layers": int(interaction_meta["n_layers"]),
                "n_segments": int(interaction_meta["n_segments"]),
                "n_nonzero_edges": int(interaction_meta["n_nonzero_edges"]),
                "n_trace_samples": int(annealing_meta["n_trace_samples"]),
                "n_annealing_trace_samples": int(annealing_meta["n_annealing_trace_samples"]),
                "n_checkpoints": int(annealing_meta["n_checkpoints"]),
                "artifact_disk_mb": dir_size_bytes(worker_root) / (1024.0 * 1024.0),
                "cpp_worker_peak_estimate_mb": estimate_cpp_worker_peak_mb(
                    n_segments=int(interaction_meta["n_segments"]),
                    n_edges=int(interaction_meta["n_nonzero_edges"]),
                    n_checkpoints=int(annealing_meta["n_checkpoints"]),
                ),
                "status": "ok",
            }
            row.update({key: native_value(value) for key, value in metrics.items()})
            row["search_score"] = compute_search_score(row)

            if args.keep_run_artifacts:
                artifact_dir = runs_root / dataset_id
                copy_worker_outputs(dataset_dir, run_dir, artifact_dir)
                row["artifact_dir"] = str(artifact_dir)

            log(
                (
                    f"[{dataset_id}] done in {row['wall_seconds']:.1f}s "
                    f"score={row['search_score']:.3f} "
                    f"eff={float(row.get('track_efficiency', 0.0)):.3f} "
                    f"fake={float(row.get('fake_rate', 0.0)):.3f}"
                )
            )
            return row
        except Exception as exc:
            failure_dir = failures_root / dataset_id
            failure_dir.mkdir(parents=True, exist_ok=True)
            error_payload = {
                "dataset_id": dataset_id,
                "slot": slot,
                "error": str(exc),
            }
            dump_json(failure_dir / "error.json", error_payload)
            if worker_root.exists():
                for src_name in ("data", "run"):
                    src_path = worker_root / src_name
                    if src_path.exists():
                        dst_path = failure_dir / src_name
                        if dst_path.exists():
                            shutil.rmtree(dst_path)
                        shutil.copytree(src_path, dst_path)
            log(f"[{dataset_id}] FAILED: {exc}")
            return {
                "dataset_id": dataset_id,
                "status": "failed",
                "error": str(exc),
            }
        finally:
            try:
                if worker_root.exists():
                    reset_dir(worker_root)
            finally:
                dataset_slots.put(slot)

    started_at = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fixed-eval") as executor:
        submitted = [executor.submit(run_dataset, idx) for idx in range(args.datasets)]
        for future in futures.as_completed(submitted):
            rows.append(future.result())
    total_wall_seconds = time.perf_counter() - started_at

    all_rows_df = pd.DataFrame(rows)
    all_rows_csv = sweep_root / "all_rows.csv"
    all_rows_df.sort_values(["status", "dataset_id"]).to_csv(all_rows_csv, index=False)

    failed_df = all_rows_df[all_rows_df["status"] != "ok"].copy()
    ok_df = all_rows_df[all_rows_df["status"] == "ok"].copy()

    if ok_df.empty:
        print("All dataset evaluations failed.", file=sys.stderr)
        return 1

    ok_df = ok_df.sort_values("dataset_id").reset_index(drop=True)
    per_dataset_csv = sweep_root / "per_dataset_metrics.csv"
    ok_df.to_csv(per_dataset_csv, index=False)

    ranked_csv = sweep_root / "ranked_by_search_score.csv"
    ok_df.sort_values("search_score", ascending=False).to_csv(ranked_csv, index=False)

    aggregate_stats = series_to_native_summary(ok_df, SUMMARY_METRICS)
    aggregate_df = pd.DataFrame(
        [
            {"metric": metric, **stats}
            for metric, stats in aggregate_stats.items()
        ]
    )
    aggregate_csv = sweep_root / "aggregate_metrics.csv"
    aggregate_df.to_csv(aggregate_csv, index=False)

    max_worker_mb = float(ok_df["cpp_worker_peak_estimate_mb"].max())
    summary_payload = {
        "created_at": stamp,
        "config_path": str(config_path),
        "output_root": str(sweep_root),
        "scratch_root": str(scratch_root),
        "datasets": int(args.datasets),
        "successful_datasets": int(len(ok_df)),
        "failed_datasets": int(len(failed_df)),
        "workers": int(workers),
        "dataset_seed_start": int(args.dataset_seed_start),
        "anneal_seed_start": int(fixed_params["anneal_seed_base"]),
        "vary_anneal_seed": bool(args.vary_anneal_seed),
        "keep_run_artifacts": bool(args.keep_run_artifacts),
        "create_plots": bool(args.create_plots),
        "total_wall_seconds": float(total_wall_seconds),
        "expected_problem_size": expected_problem_size(gen_cfg),
        "fixed_params": fixed_params,
        "aggregate_metrics": aggregate_stats,
        "rates": {
            "zero_bifurcation_rate": float((ok_df["n_bifurcations"] == 0).mean()),
            "fake_rate_below_0_10": float((ok_df["fake_rate"] <= 0.10).mean()),
            "fake_rate_below_0_20": float((ok_df["fake_rate"] <= 0.20).mean()),
            "track_efficiency_at_least_0_80": float((ok_df["track_efficiency"] >= 0.80).mean()),
            "track_efficiency_soft_at_least_0_90": float((ok_df["track_efficiency_soft"] >= 0.90).mean()),
        },
        "memory_estimate": {
            "mean_cpp_worker_peak_estimate_mb": float(ok_df["cpp_worker_peak_estimate_mb"].mean()),
            "max_cpp_worker_peak_estimate_mb": max_worker_mb,
            "recommended_total_mem_gb": int(recommend_total_mem_gb(max_worker_mb=max_worker_mb, workers=workers)),
            "note": (
                "Estimate is based on segment count, sparse-edge count, and checkpoint count. "
                "It is intentionally conservative for Slurm requests."
            ),
        },
        "artifacts": {
            "all_rows_csv": str(all_rows_csv),
            "per_dataset_csv": str(per_dataset_csv),
            "ranked_csv": str(ranked_csv),
            "aggregate_csv": str(aggregate_csv),
        },
    }
    if not failed_df.empty:
        summary_payload["artifacts"]["failures_root"] = str(failures_root)

    summary_json = sweep_root / "summary.json"
    dump_json(summary_json, summary_payload)

    summary_report = sweep_root / "summary_report.md"
    write_summary_report(summary_report, ok_df, aggregate_stats, summary_payload)

    print(f"Wrote: {per_dataset_csv}")
    print(f"Wrote: {aggregate_csv}")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {summary_report}")
    if not failed_df.empty:
        print(f"Some datasets failed. Inspect: {failures_root}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
