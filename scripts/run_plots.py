#!/usr/bin/env python3
"""
Plotting entry-point
────────────────────
Auto-detects the latest run in results/runs/ and produces:
  • hit_map.png   – detector hits (training vs. ground truth)
  • tracks.png    – reconstructed vs. ground-truth tracks

Usage
-----
    python scripts/run_plots.py              # uses latest run
    python scripts/run_plots.py <run_id>     # uses specific run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── resolve project root (works when called from repo root or scripts/) ─────
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting.plot_hits import plot_hits       # noqa: E402
from src.plotting.plot_tracks import plot_tracks   # noqa: E402

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'pyyaml'. Install dependencies first (e.g. `uv sync`)."
    ) from exc


def latest_run(results_root: Path) -> Path:
    """Return the most-recently-created run folder."""
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


def main() -> int:
    cfg_path = PROJECT_ROOT / "scripts" / "config.yaml"
    cfg = load_config(cfg_path)
    paths_cfg = cfg.get("paths", {})
    results_root = (PROJECT_ROOT / paths_cfg.get("results_root", "results/runs")).resolve()
    run_id = str(cfg.get("run", {}).get("run_id", {}))
    saved_run_dir = (results_root / run_id).resolve()

    # ── choose run ──────────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        run_dir = results_root / sys.argv[1]
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run folder not found: {run_dir}")
    else:
        run_dir = latest_run(results_root)

    print(f"Plotting run: {run_dir.name}")

    # ── resolve paths ───────────────────────────────────────────────────────
    training_csv     = (PROJECT_ROOT / paths_cfg.get("training_hits_csv", "data/training_hits.csv")).resolve()
    ground_truth_csv = (PROJECT_ROOT / "data" / "ground_truth_hits.csv").resolve()
    segments_csv     = run_dir / "interaction" / "segments.csv"
    final_state_csv  = run_dir / "annealing"   / "final_state.csv"

    for p in (training_csv, ground_truth_csv, segments_csv, final_state_csv):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    # ── detector radii from config ──────────────────────────────────────────
    detector_radii = cfg.get("generation", {}).get("data", {}).get(
        "detector_layers", [1, 2, 3, 4, 5]
    )

    # ── output directory ────────────────────────────────────────────────────
    save_dir = saved_run_dir if run_id != "auto" else run_dir
    plot_dir = (save_dir / "plots").resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── generate plots ──────────────────────────────────────────────────────
    print("Generating hit map …")
    plot_hits(
        training_csv=training_csv,
        ground_truth_csv=ground_truth_csv,
        detector_radii=detector_radii,
        out_path=plot_dir / "hit_map.png",
    )

    print("Generating track comparison …")
    plot_tracks(
        training_csv=training_csv,
        ground_truth_csv=ground_truth_csv,
        segments_csv=segments_csv,
        final_state_csv=final_state_csv,
        detector_radii=detector_radii,
        out_path=plot_dir / "tracks.png",
    )

    print(f"\nAll plots saved to: {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
