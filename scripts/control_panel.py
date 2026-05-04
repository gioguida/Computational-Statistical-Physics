#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'pyyaml'. Install dependencies first (e.g. `uv sync`)."
    ) from exc


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")

    return cfg


def resolve_run_id(run_cfg: dict[str, Any]) -> str:
    run_id = str(run_cfg.get("run_id", "auto"))
    if run_id == "auto":
        return dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_id


def resolve_switches(cfg: dict[str, Any]) -> dict[str, bool]:
    switches_cfg = cfg.get("switches", {})
    if not isinstance(switches_cfg, dict):
        raise ValueError("switches must be a mapping")

    switches = {
        "build": bool(switches_cfg.get("build", True)),
        "generate": bool(switches_cfg.get("generate", True)),
        "interaction": bool(switches_cfg.get("interaction", True)),
        "anneal": bool(switches_cfg.get("anneal", True)),
    }

    if not any(switches.values()):
        raise ValueError("All switches are false; enable at least one stage in scripts/config.yaml")

    return switches


def cmake_build(project_root: Path, build_dir: Path) -> None:
    run_cmd(["cmake", "-S", ".", "-B", str(build_dir)], cwd=project_root)
    run_cmd(["cmake", "--build", str(build_dir), "-j"], cwd=project_root)


def run_generation(project_root: Path, gen_cfg: dict[str, Any]) -> None:
    if not bool(gen_cfg.get("enabled", True)):
        print("Skipping data generation (generation.enabled=false).")
        return

    module = str(gen_cfg.get("module", "src.hits.data_gen"))
    run_cmd([sys.executable, "-m", module], cwd=project_root)


def run_interaction_stage(
    project_root: Path,
    binary: Path,
    hits_csv: Path,
    out_dir: Path,
    inter_cfg: dict[str, Any],
) -> None:
    theta_max = float(inter_cfg["theta_max"])
    merge_penalty = float(inter_cfg["merge_penalty"])
    fork_penalty = float(inter_cfg["fork_penalty"])

    run_cmd(
        [
            str(binary),
            "--hits-csv",
            str(hits_csv),
            "--out-dir",
            str(out_dir),
            "--theta-max",
            str(theta_max),
            "--merge-penalty",
            str(merge_penalty),
            "--fork-penalty",
            str(fork_penalty),
        ],
        cwd=project_root,
    )


def run_annealing_stage(
    project_root: Path,
    binary: Path,
    interaction_dir: Path,
    out_dir: Path,
    ann_cfg: dict[str, Any],
) -> None:
    segments_csv = interaction_dir / "segments.csv"
    edges_csv = interaction_dir / "J_edges.csv"

    run_cmd(
        [
            str(binary),
            "--segments-csv",
            str(segments_csv),
            "--edges-csv",
            str(edges_csv),
            "--out-dir",
            str(out_dir),
            "--t-min",
            str(float(ann_cfg["t_min"])),
            "--t-max",
            str(float(ann_cfg["t_max"])),
            "--t-step",
            str(float(ann_cfg["t_step"])),
            "--toll",
            str(float(ann_cfg.get("toll", 1e-3))),
            "--length-penalty",
            str(float(ann_cfg.get("length_penalty", 0.0))),
            "--eq-sweeps",
            str(int(ann_cfg["eq_sweeps"])),
            "--log-every-steps",
            str(int(ann_cfg.get("log_every_steps", 1))),
            "--seed",
            str(int(ann_cfg["seed"])),
        ],
        cwd=project_root,
    )

def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit(
            "This script does not accept command-line arguments. "
            "Set switches in scripts/config.yaml and run: python3 scripts/control_panel.py"
        )

    project_root = Path(__file__).resolve().parents[1]
    cfg_path = (project_root / "scripts/config.yaml").resolve()
    cfg = load_config(cfg_path)

    switches = resolve_switches(cfg)
    build_cfg = cfg.get("build", {})
    paths_cfg = cfg.get("paths", {})
    run_cfg = cfg.get("run", {})
    gen_cfg = cfg.get("generation", {})
    inter_cfg = cfg.get("interaction", {})
    ann_cfg = cfg.get("annealing", {})

    build_dir = (project_root / str(build_cfg.get("build_dir", "build"))).resolve()
    results_root = (project_root / str(paths_cfg.get("results_root", "results/runs"))).resolve()
    hits_csv = (project_root / str(paths_cfg.get("training_hits_csv", "data/training_hits.csv"))).resolve()

    run_id = resolve_run_id(run_cfg)
    run_root = results_root / run_id
    interaction_dir = run_root / "interaction"
    annealing_dir = run_root / "annealing"

    interaction_bin = build_dir / "run_interaction"
    annealing_bin = build_dir / "run_annealing"

    if switches["anneal"] and not switches["interaction"] and str(run_cfg.get("run_id", "auto")) == "auto":
        raise ValueError(
            "For anneal-only runs, set run.run_id to an existing run id in scripts/config.yaml"
        )

    if switches["build"]:
        cmake_build(project_root, build_dir)

    if switches["generate"]:
        run_generation(project_root, gen_cfg)

    if switches["interaction"] or switches["anneal"]:
        results_root.mkdir(parents=True, exist_ok=True)
        run_root.mkdir(parents=True, exist_ok=True)

    if switches["interaction"]:
        if not interaction_bin.exists():
            raise FileNotFoundError(
                f"Missing binary: {interaction_bin}. Enable switches.build and rerun control panel."
            )
        interaction_dir.mkdir(parents=True, exist_ok=True)
        run_interaction_stage(project_root, interaction_bin, hits_csv, interaction_dir, inter_cfg)

    if switches["anneal"]:
        if not annealing_bin.exists():
            raise FileNotFoundError(
                f"Missing binary: {annealing_bin}. Enable switches.build and rerun control panel."
            )
        if not (interaction_dir / "segments.csv").exists() or not (interaction_dir / "J_edges.csv").exists():
            raise FileNotFoundError(
                "Missing interaction artifacts. Enable switches.interaction or reuse an existing run.run_id."
            )
        annealing_dir.mkdir(parents=True, exist_ok=True)
        run_annealing_stage(project_root, annealing_bin, interaction_dir, annealing_dir, ann_cfg)

    print(f"Run id: {run_id}")
    print(f"Run folder: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
