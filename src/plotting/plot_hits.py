"""
Detector & Hits Visualisation
──────────────────────────────
Draws the concentric detector layers and overlays every recorded hit.
Two panels:
  • Left  – training hits coloured by detector layer
  • Right – ground-truth hits coloured by particle track
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── colour palette ──────────────────────────────────────────────────────────
_LAYER_CMAP = plt.cm.viridis
_TRACK_CMAP = plt.cm.tab20


def _detector_circles(
    ax: plt.Axes,
    radii: Sequence[float],
    color: str = "#444444",
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

    rmax = max(detector_radii) * 1.15

    fig, (ax_train, ax_truth) = plt.subplots(
        1, 2, figsize=(14, 6.5), facecolor="#0e1117"
    )

    for ax in (ax_train, ax_truth):
        ax.set_facecolor("#0e1117")
        ax.set_aspect("equal")
        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333333")
        _detector_circles(ax, detector_radii, color="#555555")

    # ── left panel: training hits, colour = layer ───────────────────────────
    n_layers = train["layer_id"].nunique()
    layer_norm = plt.Normalize(vmin=0, vmax=n_layers - 1)
    colours_layer = _LAYER_CMAP(layer_norm(train["layer_id"].values))

    ax_train.scatter(
        train["hit_x"], train["hit_y"],
        c=colours_layer, s=18, edgecolors="white", linewidths=0.3, zorder=3,
    )
    ax_train.set_title("Training hits  (colour = layer)", color="white", fontsize=11, pad=10)

    sm_layer = plt.cm.ScalarMappable(cmap=_LAYER_CMAP, norm=layer_norm)
    sm_layer.set_array([])
    cbar = fig.colorbar(sm_layer, ax=ax_train, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label("Layer", color="white", fontsize=9)
    cbar.ax.tick_params(colors="white", labelsize=7)

    # ── right panel: ground truth hits, colour = track_id ───────────────────
    n_tracks = truth["track_id"].nunique()
    track_norm = plt.Normalize(vmin=0, vmax=n_tracks - 1)
    colours_track = _TRACK_CMAP(track_norm(truth["track_id"].values))

    ax_truth.scatter(
        truth["hit_x"], truth["hit_y"],
        c=colours_track, s=18, edgecolors="white", linewidths=0.3, zorder=3,
    )
    ax_truth.set_title("Ground truth  (colour = track)", color="white", fontsize=11, pad=10)

    sm_track = plt.cm.ScalarMappable(cmap=_TRACK_CMAP, norm=track_norm)
    sm_track.set_array([])
    cbar2 = fig.colorbar(sm_track, ax=ax_truth, fraction=0.046, pad=0.04, shrink=0.85)
    cbar2.set_label("Track ID", color="white", fontsize=9)
    cbar2.ax.tick_params(colors="white", labelsize=7)

    fig.suptitle(
        "Particle Detector — Hit Map",
        color="white", fontsize=14, fontweight="bold", y=0.97,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if out_path is not None:
        fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"  ✓ saved  {out_path}")

    return fig
