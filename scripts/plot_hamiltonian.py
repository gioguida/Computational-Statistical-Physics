#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting.plot_hamiltonian import plot_hamiltonian_trace  # noqa: E402

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'pyyaml'. Install dependencies first (e.g. `uv sync`)."
    ) from exc


def latest_run(results_root: Path) -> Path:
    runs = sorted(
        [path for path in results_root.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"No run folders found in {results_root}")
    return runs[0]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> int:
    cfg = load_config(PROJECT_ROOT / "scripts" / "config.yaml")
    results_root = (
        PROJECT_ROOT / cfg.get("paths", {}).get("results_root", "results/runs")
    ).resolve()

    if len(sys.argv) > 1:
        run_dir = (results_root / sys.argv[1]).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run folder not found: {run_dir}")
    else:
        run_dir = latest_run(results_root)

    trace_csv = run_dir / "annealing" / "energy_trace.csv"
    if not trace_csv.exists():
        raise FileNotFoundError(
            f"Missing energy trace for run {run_dir.name}: {trace_csv}"
        )

    plot_dir = run_dir / "plots"
    out_path = plot_dir / "hamiltonian_trace.png"

    print(f"Plotting Hamiltonian trace for run: {run_dir.name}")
    plot_hamiltonian_trace(trace_csv=trace_csv, out_path=out_path)
    print(f"Saved plot to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
