"""
File: src/plotting/plot_hits.py
Purpose: Plotting and visualization utilities for results analysis.
Usage: Run after data/results are generated to produce figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── style palette ──────────────────────────────────────────────────────────
_FIG_BG = "#ffffff"
_AX_BG = "#ffffff"
_TEXT = "#1f1f1f"
_SPINE = "#c8c8c8"
_CIRCLE = "#bdbdbd"
_FAKE = "#d62728"
_PANEL_TITLE_FS = 13
_CBAR_LABEL_FS = 11
_CBAR_TICK_FS = 9
_SUPTITLE_FS = 17

# ── colour palette ──────────────────────────────────────────────────────────
_LAYER_CMAP = plt.cm.viridis
_TRACK_CMAP = plt.cm.tab20


def _unwrap_angles(angles: np.ndarray) -> np.ndarray:
    """Unwrap angles with minimal jumps, preserving the original ordering."""
    if angles.size == 0:
        return angles
    out = np.empty_like(angles)
    out[0] = angles[0]
    for i in range(1, angles.size):
        delta = (angles[i] - out[i - 1] + np.pi) % (2 * np.pi) - np.pi
        out[i] = out[i - 1] + delta
    return out


def _draw_ground_truth_arcs(
    ax: plt.Axes,
    truth_real: pd.DataFrame,
    track_norm: plt.Normalize,
) -> None:
    """
    Draw fitted circular trajectory arcs for each true track.

    The generation model produces circles passing through the interaction point
    (origin). For each track we fit the circle center from hit coordinates and
    draw the arc from origin to the outermost hit.
    """
    if truth_real.empty:
        return

    sort_col = "layer_radius" if "layer_radius" in truth_real.columns else "layer_id"

    for track_id, track_hits in truth_real.groupby("track_id", sort=True):
        t = track_hits.sort_values(sort_col)
        xy = t[["hit_x", "hit_y"]].to_numpy(dtype=float)
        if xy.shape[0] < 2:
            continue

        # Circle fit with origin-on-circle constraint:
        # |p - c|^2 = |c|^2  ->  2 p·c = |p|^2
        A = 2.0 * xy
        b = np.sum(xy * xy, axis=1)
        center, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < 2:
            continue

        cx, cy = float(center[0]), float(center[1])
        radius = float(np.hypot(cx, cy))
        if radius <= 0:
            continue

        theta_hits = np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)
        theta_hits = _unwrap_angles(theta_hits)

        # Include origin so the arc represents the trajectory itself, not only
        # the detector-crossing segment.
        theta_origin = float(np.arctan2(-cy, -cx))
        delta = (theta_hits[0] - theta_origin + np.pi) % (2 * np.pi) - np.pi
        theta_start = theta_hits[0] - delta
        theta_end = theta_hits[-1]

        arc_theta = np.linspace(theta_start, theta_end, 180)
        arc_x = cx + radius * np.cos(arc_theta)
        arc_y = cy + radius * np.sin(arc_theta)

        c = _TRACK_CMAP(track_norm(track_id))
        ax.plot(arc_x, arc_y, color=c, lw=1.0, alpha=0.7, zorder=2)


def _detector_circles(
    ax: plt.Axes,
    radii: Sequence[float],
    color: str = _CIRCLE,
    ls: str = "--",
    lw: float = 0.8,
) -> None:
    """Draw thin dashed circles for each detector layer."""
    theta = np.linspace(0, 2 * np.pi, 300)
    for r in radii:
        ax.plot(r * np.cos(theta), r * np.sin(theta), color=color, ls=ls, lw=lw)


def plot_hits(
    training_csv: Path,
    ground_truth_csv: Path,
    detector_radii: Sequence[float],
    out_path: Optional[Path] = None,
    dpi: int = 180,
) -> plt.Figure:
    """
    Create a two-panel figure of detector hits.

    Parameters
    ----------
    training_csv : Path to training_hits.csv
    ground_truth_csv : Path to ground_truth_hits.csv
    detector_radii : list of detector layer radii (for drawing circles)
    out_path : if given, save the figure there
    dpi : resolution
    """
    train = pd.read_csv(training_csv)
    truth = pd.read_csv(ground_truth_csv)
    real_hit_ids = set(truth["hit_id"].tolist())
    fake_train = train.loc[~train["hit_id"].isin(real_hit_ids)].copy()
    real_train = train.loc[train["hit_id"].isin(real_hit_ids)].copy()

    rmax = max(detector_radii) * 1.15

    fig, (ax_train, ax_truth) = plt.subplots(
        1, 2, figsize=(14, 6.5), facecolor=_FIG_BG
    )

    for ax in (ax_train, ax_truth):
        ax.set_facecolor(_AX_BG)
        ax.set_aspect("equal")
        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_color(_SPINE)
        _detector_circles(ax, detector_radii, color=_CIRCLE)

    # ── left panel: training hits, colour = layer ───────────────────────────
    n_layers = real_train["layer_id"].nunique() if not real_train.empty else train["layer_id"].nunique()
    layer_norm = plt.Normalize(vmin=0, vmax=max(n_layers - 1, 1))
    colours_layer = _LAYER_CMAP(layer_norm(real_train["layer_id"].values))

    if not real_train.empty:
        ax_train.scatter(
            real_train["hit_x"], real_train["hit_y"],
            c=colours_layer, s=20, edgecolors="#222222", linewidths=0.35, zorder=3,
        )
    if not fake_train.empty:
        ax_train.scatter(
            fake_train["hit_x"], fake_train["hit_y"],
            marker="x", s=38, color=_FAKE, linewidths=1.4, zorder=4,
        )
    ax_train.set_title(
        f"Training hits  (colour = layer, fake x = {len(fake_train)})",
        color=_TEXT,
        fontsize=_PANEL_TITLE_FS,
        pad=10,
    )

    sm_layer = plt.cm.ScalarMappable(cmap=_LAYER_CMAP, norm=layer_norm)
    sm_layer.set_array([])
    cbar = fig.colorbar(sm_layer, ax=ax_train, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label("Layer", color=_TEXT, fontsize=_CBAR_LABEL_FS)
    cbar.ax.tick_params(colors=_TEXT, labelsize=_CBAR_TICK_FS)
    cbar.outline.set_edgecolor(_SPINE)

    # ── right panel: ground truth hits, colour = track_id ───────────────────
    truth_real = truth.loc[truth["track_id"] >= 0].copy()
    truth_fake = truth.loc[truth["track_id"] < 0].copy()
    n_tracks = truth_real["track_id"].nunique()
    track_norm = plt.Normalize(vmin=0, vmax=max(n_tracks - 1, 1))
    _draw_ground_truth_arcs(ax_truth, truth_real, track_norm)
    if not truth_real.empty:
        colours_track = _TRACK_CMAP(track_norm(truth_real["track_id"].values))
        ax_truth.scatter(
            truth_real["hit_x"], truth_real["hit_y"],
            c=colours_track, s=20, edgecolors="#222222", linewidths=0.35, zorder=3,
        )
    if not truth_fake.empty:
        ax_truth.scatter(
            truth_fake["hit_x"], truth_fake["hit_y"],
            marker="x", s=38, color=_FAKE, linewidths=1.4, zorder=4,
        )
    ax_truth.set_title(
        f"Ground truth  (colour = track, fake x = {len(truth_fake)})",
        color=_TEXT,
        fontsize=_PANEL_TITLE_FS,
        pad=10,
    )

    if n_tracks > 0:
        sm_track = plt.cm.ScalarMappable(cmap=_TRACK_CMAP, norm=track_norm)
        sm_track.set_array([])
        cbar2 = fig.colorbar(sm_track, ax=ax_truth, fraction=0.046, pad=0.04, shrink=0.85)
        cbar2.set_label("Track ID", color=_TEXT, fontsize=_CBAR_LABEL_FS)
        cbar2.ax.tick_params(colors=_TEXT, labelsize=_CBAR_TICK_FS)
        cbar2.outline.set_edgecolor(_SPINE)

    fig.suptitle(
        "Particle Detector — Hit Map",
        color=_TEXT, fontsize=_SUPTITLE_FS, fontweight="bold", y=0.97,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if out_path is not None:
        fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"  saved  {out_path}")

    return fig
