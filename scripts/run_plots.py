#!/usr/bin/env python3
"""
Plotting entry-point.

Usage:
    python scripts/run_plots.py
    python scripts/run_plots.py <run_id>
    python scripts/run_plots.py --run-dir results/ds_001_trial_0083
    python scripts/run_plots.py --state lowest
    python scripts/run_plots.py <run_id> --state lowest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting.plot_hits import plot_hits
from src.plotting.plot_hamiltonian import plot_hamiltonian_trace
from src.plotting.plot_tracks import plot_tracks
from src.plotting.plot_annealing_animation import plot_annealing_state_animation

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'pyyaml'. Install dependencies first (e.g. `uv sync`)."
    ) from exc


def latest_run(results_root: Path) -> Path:
    runs = sorted(
        [p for p in results_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No run folders found in {results_root}")
    return runs[0]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def csv_row_count(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def parse_cli_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots for a run folder")
    parser.add_argument("run_id", nargs="?", help="Run id under results/runs")
    parser.add_argument("--state", default="final", help="Which annealing state to plot: final or lowest")
    parser.add_argument(
        "--run-dir",
        help="Explicit path to a run folder containing interaction/ and annealing/",
    )
    parser.add_argument(
        "--training-csv",
        help="Explicit path to training_hits.csv (needed for hit/track plots)",
    )
    parser.add_argument(
        "--ground-truth-csv",
        help="Explicit path to ground_truth_hits.csv (needed for hit/track plots)",
    )
    args = parser.parse_args(argv)

    args.state = str(args.state).strip().lower()
    if args.state not in ("final", "lowest"):
        raise ValueError(f"Invalid --state value '{args.state}'; expected 'final' or 'lowest'")
    if args.run_id is not None and args.run_dir is not None:
        raise ValueError("Use either positional run_id or --run-dir, not both")
    return args


def main() -> int:
    cfg_path = PROJECT_ROOT / "scripts" / "config.yaml"
    cfg = load_config(cfg_path)
    paths_cfg = cfg.get("paths", {})
    results_root = (PROJECT_ROOT / paths_cfg.get("results_root", "results/runs")).resolve()
    configured_run_id = str(cfg.get("run", {}).get("run_id", {}))
    saved_run_dir = (results_root / configured_run_id).resolve()

    args = parse_cli_args(sys.argv[1:])
    state_source = args.state

    if args.run_dir is not None:
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run folder not found: {run_dir}")
    elif args.run_id is not None:
        run_dir = results_root / args.run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run folder not found: {run_dir}")
    else:
        run_dir = latest_run(results_root)

    print(f"Plotting run: {run_dir.name}")

    training_csv = (
        Path(args.training_csv).resolve()
        if args.training_csv is not None
        else (PROJECT_ROOT / paths_cfg.get("training_hits_csv", "data/training_hits.csv")).resolve()
    )
    ground_truth_csv = (
        Path(args.ground_truth_csv).resolve()
        if args.ground_truth_csv is not None
        else (PROJECT_ROOT / "data" / "ground_truth_hits.csv").resolve()
    )
    segments_csv = run_dir / "interaction" / "segments.csv"
    interaction_meta_json = run_dir / "interaction" / "interaction_meta.json"
    state_filename = "final_state.csv" if state_source == "final" else "lowest_energy_state.csv"
    final_state_csv = run_dir / "annealing" / state_filename
    energy_trace_csv = run_dir / "annealing" / "energy_trace.csv"

    for p in (segments_csv, final_state_csv):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    detector_radii = cfg.get("generation", {}).get("data", {}).get(
        "detector_layers", [1, 2, 3, 4, 5]
    )
    plotting_cfg = cfg.get("plotting", {})
    create_animation = bool(plotting_cfg.get("create_annealing_animation", False))
    animation_fps = int(plotting_cfg.get("annealing_animation_fps", 8))

    save_dir = (
        saved_run_dir
        if configured_run_id != "auto" and args.run_dir is None and args.run_id is None
        else run_dir
    )
    plot_dir = (save_dir / "plots").resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    can_plot_hit_and_tracks = training_csv.exists() and ground_truth_csv.exists()
    if can_plot_hit_and_tracks and args.run_dir is not None and args.training_csv is None:
        if interaction_meta_json.exists():
            interaction_meta = load_json(interaction_meta_json)
            expected_n_hits = int(interaction_meta.get("n_hits", -1))
            actual_n_hits = csv_row_count(training_csv)
            if expected_n_hits >= 0 and actual_n_hits != expected_n_hits:
                can_plot_hit_and_tracks = False
                print("Skipping hit map and track comparison: local training CSV does not match this run.")
                print(f"Run metadata n_hits:       {expected_n_hits}")
                print(f"Provided training CSV rows: {actual_n_hits}")

    if can_plot_hit_and_tracks:
        print("Generating hit map...")
        plot_hits(
            training_csv=training_csv,
            ground_truth_csv=ground_truth_csv,
            detector_radii=detector_radii,
            out_path=plot_dir / "hit_map.png",
        )

        print("Generating track comparison...")
        print(f"Using annealing state: {state_filename}")
        plot_tracks(
            training_csv=training_csv,
            ground_truth_csv=ground_truth_csv,
            segments_csv=segments_csv,
            final_state_csv=final_state_csv,
            detector_radii=detector_radii,
            out_path=plot_dir / "tracks.png",
        )
    else:
        if not training_csv.exists() or not ground_truth_csv.exists():
            print("Skipping hit map and track comparison: training/ground-truth CSVs not available.")
            print(f"Expected training CSV:     {training_csv}")
            print(f"Expected ground-truth CSV: {ground_truth_csv}")

    if energy_trace_csv.exists():
        print("Generating Hamiltonian trace plot...")
        plot_hamiltonian_trace(
            trace_csv=energy_trace_csv,
            out_path=plot_dir / "hamiltonian_trace.png",
        )
    else:
        print("Skipping Hamiltonian trace plot: missing energy_trace.csv")

    if create_animation:
        checkpoints_csv = run_dir / "annealing" / "state_checkpoints.csv"
        if checkpoints_csv.exists():
            print("Generating annealing state animation...")
            plot_annealing_state_animation(
                training_csv=training_csv,
                ground_truth_csv=ground_truth_csv,
                segments_csv=segments_csv,
                checkpoints_csv=checkpoints_csv,
                detector_radii=detector_radii,
                out_path=plot_dir / "annealing_state_evolution.gif",
                fps=animation_fps,
            )
        else:
            print("Skipping annealing state animation: missing state_checkpoints.csv")

    print(f"\nAll plots saved to: {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
