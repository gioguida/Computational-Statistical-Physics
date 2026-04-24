"""
Track Reconstruction Visualisation
────────────────────────────────────
Side-by-side comparison:
  • Left  – ground-truth tracks (hits connected by track_id)
  • Right – reconstructed tracks (selected segments from simulated annealing)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_TRACK_CMAP = plt.cm.tab20


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


def _setup_ax(ax: plt.Axes, rmax: float) -> None:
    ax.set_facecolor("#0e1117")
    ax.set_aspect("equal")
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#333333")


# ── ground-truth drawing ────────────────────────────────────────────────────

def _draw_ground_truth(
    ax: plt.Axes,
    truth: pd.DataFrame,
    detector_radii: Sequence[float],
) -> None:
    """Connect hits belonging to the same track across layers."""
    n_tracks = truth["track_id"].nunique()
    track_ids = sorted(truth["track_id"].unique())
    norm = plt.Normalize(vmin=0, vmax=max(n_tracks - 1, 1))

    _detector_circles(ax, detector_radii, color="#555555")

    for tid in track_ids:
        t = truth[truth["track_id"] == tid].sort_values("layer_id")
        c = _TRACK_CMAP(norm(tid))
        ax.plot(t["hit_x"], t["hit_y"], "-", color=c, lw=1.2, alpha=0.85, zorder=2)
        ax.scatter(t["hit_x"], t["hit_y"], color=c, s=16, edgecolors="white",
                   linewidths=0.3, zorder=3)


# ── reconstruction drawing ──────────────────────────────────────────────────

def _draw_reconstructed(
    ax: plt.Axes,
    segments: pd.DataFrame,
    final_state: pd.DataFrame,
    hits: pd.DataFrame,
    detector_radii: Sequence[float],
) -> int:
    """
    Draw segments selected by the annealing (selected == 1).
    Returns the number of reconstructed track chains found.
    """
    selected = final_state[final_state["selected"] == 1]
    sel_segs = segments.merge(selected[["seg_id"]], on="seg_id")

    # Build a lookup: hit_id → (x, y)
    hit_pos = hits.set_index("hit_id")[["hit_x", "hit_y"]]

    _detector_circles(ax, detector_radii, color="#555555")

    # ── chain selected segments into tracks ──────────────────────────────
    # A chain is a maximal sequence of segments where one's hit_b == next's hit_a
    # Build adjacency: hit_b → segment starting at that hit_b
    seg_list = sel_segs.to_dict("records")
    outgoing = {}  # hit_b → seg  (from layer perspective: hit that ends one seg starts the next)
    for seg in seg_list:
        outgoing.setdefault(seg["hit_a"], []).append(seg)

    # Find chain heads: segments whose hit_a is NOT the hit_b of any other selected segment
    all_hit_b = set(sel_segs["hit_b"])
    heads = [s for s in seg_list if s["hit_a"] not in all_hit_b]

    # If no clear heads (rare), just start from lowest-layer segments
    if not heads:
        min_layer = sel_segs["layer_a"].min()
        heads = [s for s in seg_list if s["layer_a"] == min_layer]

    visited_segs = set()
    chains: list[list[dict]] = []

    for head in heads:
        if head["seg_id"] in visited_segs:
            continue
        chain = [head]
        visited_segs.add(head["seg_id"])
        current = head
        while True:
            nexts = [s for s in outgoing.get(current["hit_b"], [])
                     if s["seg_id"] not in visited_segs]
            if not nexts:
                break
            nxt = nexts[0]
            chain.append(nxt)
            visited_segs.add(nxt["seg_id"])
            current = nxt
        chains.append(chain)

    # Also pick up any orphan segments not yet visited
    for seg in seg_list:
        if seg["seg_id"] not in visited_segs:
            chains.append([seg])
            visited_segs.add(seg["seg_id"])

    n_chains = len(chains)
    norm = plt.Normalize(vmin=0, vmax=max(n_chains - 1, 1))

    for i, chain in enumerate(chains):
        c = _TRACK_CMAP(norm(i))
        for seg in chain:
            ha, hb = seg["hit_a"], seg["hit_b"]
            if ha in hit_pos.index and hb in hit_pos.index:
                xa, ya = hit_pos.loc[ha]
                xb, yb = hit_pos.loc[hb]
                ax.plot([xa, xb], [ya, yb], "-", color=c, lw=1.2, alpha=0.85, zorder=2)
                ax.scatter([xa, xb], [ya, yb], color=c, s=16, edgecolors="white",
                           linewidths=0.3, zorder=3)

    return n_chains


# ── public API ──────────────────────────────────────────────────────────────

def plot_tracks(
    training_csv: Path,
    ground_truth_csv: Path,
    segments_csv: Path,
    final_state_csv: Path,
    detector_radii: Sequence[float],
    out_path: Optional[Path] = None,
    dpi: int = 180,
) -> plt.Figure:
    """
    Two-panel figure: ground truth tracks vs. reconstructed tracks.

    Parameters
    ----------
    training_csv      : path to training_hits.csv (hit coordinates)
    ground_truth_csv  : path to ground_truth_hits.csv (hits with track_id)
    segments_csv      : path to <run>/interaction/segments.csv
    final_state_csv   : path to <run>/annealing/final_state.csv
    detector_radii    : list of detector layer radii
    out_path          : if given, save the figure
    dpi               : resolution
    """
    hits   = pd.read_csv(training_csv)
    truth  = pd.read_csv(ground_truth_csv)
    segs   = pd.read_csv(segments_csv)
    state  = pd.read_csv(final_state_csv)

    rmax = max(detector_radii) * 1.15

    fig, (ax_gt, ax_reco) = plt.subplots(
        1, 2, figsize=(14, 6.5), facecolor="#0e1117",
    )
    _setup_ax(ax_gt, rmax)
    _setup_ax(ax_reco, rmax)

    # ── left: ground truth ──────────────────────────────────────────────────
    _draw_ground_truth(ax_gt, truth, detector_radii)
    n_true = truth["track_id"].nunique()
    ax_gt.set_title(
        f"Ground truth  ({n_true} tracks)",
        color="white", fontsize=11, pad=10,
    )

    # ── right: reconstruction ───────────────────────────────────────────────
    n_reco = _draw_reconstructed(ax_reco, segs, state, hits, detector_radii)
    n_sel = int((state["selected"] == 1).sum())
    ax_reco.set_title(
        f"Reconstructed  ({n_reco} chains, {n_sel} segments)",
        color="white", fontsize=11, pad=10,
    )

    fig.suptitle(
        "Track Reconstruction — Ground Truth vs. Annealing",
        color="white", fontsize=14, fontweight="bold", y=0.97,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if out_path is not None:
        fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"  ✓ saved  {out_path}")

    return fig
