#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'pyyaml'. Install dependencies first (e.g. `uv sync`)."
    ) from exc

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting.metrics import visualize_metrics


def latest_run(results_root: Path) -> Path:
    runs = sorted([p for p in results_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run folders found in {results_root}")
    return runs[0]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    cfg_path = PROJECT_ROOT / "scripts" / "config.yaml"
    cfg = load_config(cfg_path)

    paths_cfg = cfg.get("paths", {})
    results_root = (PROJECT_ROOT / str(paths_cfg.get("results_root", "results/runs"))).resolve()
    run_cfg = cfg.get("run", {})
    run_id_cfg = str(run_cfg.get("run_id", "auto"))

    cli_run_id = sys.argv[1].strip() if len(sys.argv) > 1 else None
    if len(sys.argv) > 2:
        raise ValueError("Usage: python scripts/run_metrics.py [run_id]")

    if cli_run_id:
        run_dir = (results_root / cli_run_id).resolve()
    elif run_id_cfg != "auto":
        run_dir = (results_root / run_id_cfg).resolve()
    else:
        run_dir = latest_run(results_root).resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")

    detector_layers = cfg.get("generation", {}).get("data", {}).get("detector_layers", [1, 2, 3, 4, 5])
    if not isinstance(detector_layers, list) or len(detector_layers) < 2:
        raise ValueError("generation.data.detector_layers must contain at least two radii")

    metrics_cfg = {
        "project_root": str(PROJECT_ROOT),
        "run_dir": str(run_dir),
        "training_hits_csv": str((PROJECT_ROOT / str(paths_cfg.get("training_hits_csv", "data/training_hits.csv"))).resolve()),
        "ground_truth_csv": str((PROJECT_ROOT / str(paths_cfg.get("ground_truth_hits_csv", "data/ground_truth_hits.csv"))).resolve()),
        "n_layers": len(detector_layers),
        "merge_penalty": float(cfg.get("interaction", {}).get("merge_penalty", 0.0)),
        "fork_penalty": float(cfg.get("interaction", {}).get("fork_penalty", 0.0)),
        "angle_penalty": float(cfg.get("interaction", {}).get("angle_penalty", 0.0)),
    }

    print(f"Computing metrics for run: {run_dir.name}")
    visualize_metrics(metrics_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
