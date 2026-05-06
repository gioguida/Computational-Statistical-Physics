from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


TRACE_COLUMNS = (
    "step",
    "T",
    "H_mean",
    "H_var",
    "C_v",
    "acceptance_rate",
    "H_min_so_far",
    "n_active_mean",
    "n_active_std",
    "delta_E_mean_neg",
    "delta_E_mean_pos",
)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _load_trace(trace_csv: Path) -> pd.DataFrame:
    trace = pd.read_csv(trace_csv)
    missing = [column for column in TRACE_COLUMNS if column not in trace.columns]
    if missing:
        raise ValueError(f"Annealing trace missing required columns {missing}: {trace_csv}")
    for column in TRACE_COLUMNS:
        trace[column] = pd.to_numeric(trace[column], errors="raise")
    if trace.empty:
        raise ValueError(f"Annealing trace is empty: {trace_csv}")
    return trace


def _load_meta(meta_json: Path) -> dict[str, Any]:
    with meta_json.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_segment_truth(segments: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    truth_tracks = truth.loc[:, ["hit_id", "track_id"]].rename(columns={"track_id": "track_id_a"})
    segment_truth = segments.merge(truth_tracks, left_on="hit_a", right_on="hit_id", how="left").drop(columns=["hit_id"])
    truth_tracks = truth.loc[:, ["hit_id", "track_id"]].rename(columns={"track_id": "track_id_b"})
    segment_truth = segment_truth.merge(truth_tracks, left_on="hit_b", right_on="hit_id", how="left").drop(columns=["hit_id"])
    segment_truth["is_true_segment"] = (
        (segment_truth["track_id_a"] == segment_truth["track_id_b"])
        & (segment_truth["track_id_a"] != -1)
    )
    return segment_truth


def _compute_track_efficiencies(
    active_segment_keys: set[tuple[int, int]],
    truth: pd.DataFrame,
    n_layers: int,
) -> tuple[float, float]:
    if n_layers <= 1:
        return 0.0, 0.0

    truth_real = truth[truth["track_id"] != -1].copy()
    if truth_real.empty:
        return 0.0, 0.0

    strict_hits_required = n_layers - 1
    soft_hits_required = max(n_layers - 2, 0)
    track_ids = sorted(truth_real["track_id"].unique())
    strict_ok = 0
    soft_ok = 0

    for track_id in track_ids:
        track_hits = truth_real[truth_real["track_id"] == track_id].sort_values("layer_id")
        active_count = 0
        for i in range(len(track_hits) - 1):
            row_a = track_hits.iloc[i]
            row_b = track_hits.iloc[i + 1]
            if int(row_b["layer_id"]) - int(row_a["layer_id"]) != 1:
                continue
            key = (int(row_a["hit_id"]), int(row_b["hit_id"]))
            if key in active_segment_keys:
                active_count += 1

        if active_count >= strict_hits_required:
            strict_ok += 1
        if active_count >= soft_hits_required:
            soft_ok += 1

    denom = len(track_ids)
    return _safe_div(strict_ok, denom), _safe_div(soft_ok, denom)


def _compute_bifurcation_count(active_segments: pd.DataFrame) -> int:
    outgoing = active_segments.groupby("hit_a").size()
    incoming = active_segments.groupby("hit_b").size()
    all_hits = set(outgoing.index.tolist()) | set(incoming.index.tolist())
    bifurcations = 0
    for hit_id in all_hits:
        n_out = int(outgoing.get(hit_id, 0))
        n_in = int(incoming.get(hit_id, 0))
        if n_out > 1 or n_in > 1:
            bifurcations += 1
    return bifurcations


def _compute_energy_decomposition(edges: pd.DataFrame, spins: pd.Series) -> tuple[float, float, float]:
    spin_map = spins.to_dict()
    h_alignment = 0.0
    h_competing = 0.0
    h_total = 0.0

    for _, row in edges.iterrows():
        i = int(row["i"])
        j = int(row["j"])
        jij = float(row["Jij"])
        if i not in spin_map or j not in spin_map:
            raise ValueError(f"Missing spin value for edge ({i}, {j})")
        contrib = -jij * float(spin_map[i]) * float(spin_map[j])
        h_total += contrib
        if jij > 0:
            h_alignment += contrib
        elif jij < 0:
            h_competing += contrib

    return h_total, h_alignment, h_competing


def _plot_cv_vs_t(trace: pd.DataFrame, out_path: Path) -> float:
    idx_peak = int(trace["C_v"].idxmax())
    freeze_temperature = float(trace.loc[idx_peak, "T"])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(trace["T"], trace["C_v"], marker="o", linewidth=1.4, markersize=3)
    ax.scatter([freeze_temperature], [trace.loc[idx_peak, "C_v"]], color="red", s=35, zorder=3)
    ax.set_xlabel("T")
    ax.set_ylabel("C_v")
    ax.set_title("Specific Heat vs Temperature")
    ax.grid(alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return freeze_temperature


def _plot_acceptance_vs_t(trace: pd.DataFrame, out_path: Path) -> bool:
    # Flag if acceptance falls below 0.01 while additional lower energies are still reached later.
    future_min = trace["H_min_so_far"][::-1].cummin()[::-1]
    low_acc_mask = trace["acceptance_rate"] < 0.01
    still_declining_mask = future_min < (trace["H_min_so_far"] - 1e-12)
    early_freeze_flag = bool((low_acc_mask & still_declining_mask).any())

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(trace["T"], trace["acceptance_rate"], marker="o", linewidth=1.4, markersize=3)
    ax.axhline(0.01, color="red", linestyle="--", linewidth=1.0, label="0.01 threshold")
    ax.set_xlabel("T")
    ax.set_ylabel("acceptance_rate")
    ax.set_title("Acceptance Rate vs Temperature")
    ax.grid(alpha=0.3)
    ax.invert_xaxis()
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return early_freeze_flag


def _plot_energy_trace(trace: pd.DataFrame, out_path: Path) -> bool:
    n_rows = len(trace)
    idx_min = int(trace["H_min_so_far"].idxmin())
    late_threshold = int(0.9 * n_rows)
    late_minimum = idx_min >= late_threshold

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(trace["step"], trace["H_mean"], label="H_mean", linewidth=1.5)
    ax.plot(trace["step"], trace["H_min_so_far"], label="H_min_so_far", linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("Energy")
    ax.set_title("Annealing Energy Trace")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return late_minimum


def visualize_metrics(cfg: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(cfg["project_root"]).resolve()
    run_dir = Path(cfg["run_dir"]).resolve()

    interaction_dir = run_dir / "interaction"
    annealing_dir = run_dir / "annealing"
    plot_dir = run_dir / "plots"

    hits_csv = Path(cfg["training_hits_csv"]).resolve()
    truth_csv = Path(cfg["ground_truth_csv"]).resolve()
    segments_csv = interaction_dir / "segments.csv"
    edges_csv = interaction_dir / "J_edges.csv"
    final_state_csv = annealing_dir / "final_state.csv"
    trace_csv = annealing_dir / "annealing_trace.csv"
    meta_json = annealing_dir / "annealing_meta.json"

    for path in (hits_csv, truth_csv, segments_csv, edges_csv, final_state_csv, trace_csv, meta_json):
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    _ = project_root  # carried for explicit API shape and future extension
    hits = pd.read_csv(hits_csv)
    truth = pd.read_csv(truth_csv)
    segments = pd.read_csv(segments_csv)
    final_state = pd.read_csv(final_state_csv)
    trace = _load_trace(trace_csv)
    edges = pd.read_csv(edges_csv)
    meta = _load_meta(meta_json)

    # Segment-level metrics
    segment_truth = _build_segment_truth(segments, truth)
    active = final_state[final_state["selected"] == 1].copy()
    active_segments = segments.merge(active[["seg_id", "spin", "selected"]], on="seg_id", how="inner")
    active_truth = segment_truth.merge(active[["seg_id"]], on="seg_id", how="inner")

    n_true_segments_total = int(segment_truth["is_true_segment"].sum())
    n_active_segments_total = int(len(active_segments))
    n_active_true_segments = int(active_truth["is_true_segment"].sum())

    tpr = _safe_div(n_active_true_segments, n_true_segments_total)
    precision = _safe_div(n_active_true_segments, n_active_segments_total)
    f1 = _safe_div(2.0 * precision * tpr, precision + tpr)

    # Track-level metrics
    active_segment_keys = set(zip(active_segments["hit_a"].astype(int), active_segments["hit_b"].astype(int)))
    n_layers = int(cfg["n_layers"])
    track_eff, track_eff_soft = _compute_track_efficiencies(active_segment_keys, truth, n_layers)

    n_active_fake_segments = n_active_segments_total - n_active_true_segments
    fake_rate = _safe_div(n_active_fake_segments, n_active_segments_total)
    n_bifurcations = _compute_bifurcation_count(active_segments)

    # Energy decomposition
    h_final, h_alignment, h_competing = _compute_energy_decomposition(
        edges=edges,
        spins=final_state.set_index("seg_id")["spin"],
    )

    # Annealing quality indicators
    freeze_temperature = _plot_cv_vs_t(trace, plot_dir / "cv_vs_T.png")
    low_acceptance_while_declining = _plot_acceptance_vs_t(trace, plot_dir / "acceptance_vs_T.png")
    late_minimum = _plot_energy_trace(trace, plot_dir / "energy_trace.png")

    if low_acceptance_while_declining:
        print("Warning: acceptance_rate dropped below 0.01 while lower energies were still reached later.")
    if late_minimum:
        print("Warning: H_min_so_far was reached in the last 10% of steps; consider more steps or lower t_min.")

    summary = {
        "T_max": float(meta.get("t_max", trace["T"].max())),
        "T_min": float(meta.get("t_min", trace["T"].min())),
        "n_steps": int(meta.get("n_steps", len(trace))),
        "TPR": float(tpr),
        "precision": float(precision),
        "F1": float(f1),
        "track_efficiency": float(track_eff),
        "track_efficiency_soft": float(track_eff_soft),
        "fake_rate": float(fake_rate),
        "n_bifurcations": int(n_bifurcations),
        "H_final": float(h_final),
        "H_alignment": float(h_alignment),
        "H_competing": float(h_competing),
        "freeze_temperature": float(freeze_temperature),
        "length_penalty": float(meta.get("length_penalty", 0.0)),
        "merge_penalty": float(cfg["merge_penalty"]),
        "fork_penalty": float(cfg["fork_penalty"]),
        "angle_penalty": float(cfg["angle_penalty"]),
        "layer_radius_penalty": float(meta.get("layer_radius_penalty", 0.0)),
    }

    metrics_path = annealing_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    print("\nMetrics summary:")
    for key in summary.keys():
        print(f"{key:22}:: {summary[key]}")
    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved plots to:   {plot_dir}")

    return summary
