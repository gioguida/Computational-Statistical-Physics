from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


def _detector_circles(
    ax: plt.Axes,
    radii: Sequence[float],
    color: str = "#444444",
    ls: str = "--",
    lw: float = 0.8,
) -> None:
    theta = np.linspace(0, 2 * np.pi, 300)
    for r in radii:
        ax.plot(r * np.cos(theta), r * np.sin(theta), color=color, ls=ls, lw=lw)


def _load_checkpoints(checkpoints_csv: Path) -> tuple[pd.DataFrame, np.ndarray]:
    if not checkpoints_csv.exists():
        raise FileNotFoundError(f"Checkpoint CSV not found: {checkpoints_csv}")

    checkpoints = pd.read_csv(checkpoints_csv)
    required = ("checkpoint_idx", "step", "temperature", "energy")
    missing = [col for col in required if col not in checkpoints.columns]
    if missing:
        raise ValueError(f"Checkpoint CSV is missing required columns {missing}: {checkpoints_csv}")

    spin_columns = [col for col in checkpoints.columns if col.startswith("spin_")]
    if not spin_columns:
        raise ValueError(f"Checkpoint CSV has no spin columns: {checkpoints_csv}")

    spins = checkpoints[spin_columns].to_numpy(dtype=int)
    return checkpoints, spins


def _build_segment_lines(segments: pd.DataFrame, hits: pd.DataFrame) -> np.ndarray:
    hit_pos = hits.set_index("hit_id")[["hit_x", "hit_y"]]
    segs = segments.sort_values("seg_id").reset_index(drop=True)

    lines = np.empty((len(segs), 2, 2), dtype=float)
    for i, seg in segs.iterrows():
        ha = int(seg["hit_a"])
        hb = int(seg["hit_b"])
        if ha not in hit_pos.index or hb not in hit_pos.index:
            raise ValueError(f"Segment references unknown hit ids: seg_id={seg['seg_id']}")
        xa, ya = hit_pos.loc[ha]
        xb, yb = hit_pos.loc[hb]
        lines[i] = np.array([[xa, ya], [xb, yb]], dtype=float)

    return lines


def plot_annealing_state_animation(
    training_csv: Path,
    ground_truth_csv: Path,
    segments_csv: Path,
    checkpoints_csv: Path,
    detector_radii: Sequence[float],
    out_path: Optional[Path] = None,
    fps: int = 8,
    dpi: int = 120,
) -> None:
    hits = pd.read_csv(training_csv)
    truth = pd.read_csv(ground_truth_csv)
    segments = pd.read_csv(segments_csv)
    checkpoints, spins = _load_checkpoints(checkpoints_csv)
    segment_lines = _build_segment_lines(segments, hits)
    real_hit_ids = set(truth["hit_id"].tolist())
    fake_hits = hits.loc[~hits["hit_id"].isin(real_hit_ids)].copy()

    if spins.shape[1] != len(segment_lines):
        raise ValueError(
            f"Spin count mismatch: checkpoints have {spins.shape[1]} spins, "
            f"segments.csv has {len(segment_lines)} segments"
        )

    rmax = max(detector_radii) * 1.15
    fig, ax = plt.subplots(figsize=(7.5, 7.0), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.set_aspect("equal")
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    _detector_circles(ax, detector_radii, color="#555555")
    ax.scatter(hits["hit_x"], hits["hit_y"], s=9, color="#9ba3af", alpha=0.35, zorder=1)
    if not fake_hits.empty:
        ax.scatter(
            fake_hits["hit_x"],
            fake_hits["hit_y"],
            marker="x",
            s=28,
            color="#ff4d6d",
            linewidths=1.1,
            alpha=0.95,
            zorder=2,
        )

    selected_collection = LineCollection([], colors="#4ecdc4", linewidths=1.4, alpha=0.95, zorder=3)
    ax.add_collection(selected_collection)

    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "#161b22", "ec": "#49556a", "alpha": 0.95},
    )
    ax.set_title("Annealing State Evolution", color="white", fontsize=12, pad=10)

    def _update(frame_idx: int):
        spin_row = spins[frame_idx]
        selected_mask = spin_row > 0
        selected_collection.set_segments(segment_lines[selected_mask])

        row = checkpoints.iloc[frame_idx]
        n_selected = int(selected_mask.sum())
        info_text.set_text(
            f"checkpoint {int(row['checkpoint_idx'])}\n"
            f"step {int(row['step'])}, T={float(row['temperature']):.4f}\n"
            f"H={float(row['energy']):.4f}, selected={n_selected}, fake={len(fake_hits)}"
        )
        return selected_collection, info_text

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(checkpoints),
        interval=max(1, int(1000 / max(1, fps))),
        blit=False,
        repeat=True,
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = animation.PillowWriter(fps=max(1, fps))
        ani.save(out_path, writer=writer, dpi=dpi)
        print(f"  saved  {out_path}")
