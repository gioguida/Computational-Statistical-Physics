from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = ("step", "temperature", "energy", "n_selected")


def load_energy_trace(trace_csv: Path) -> pd.DataFrame:
    if not trace_csv.exists():
        raise FileNotFoundError(f"Energy trace CSV not found: {trace_csv}")

    trace = pd.read_csv(trace_csv)
    missing = [column for column in REQUIRED_COLUMNS if column not in trace.columns]
    if missing:
        raise ValueError(
            f"Energy trace is missing required columns {missing}: {trace_csv}"
        )
    if trace.empty:
        raise ValueError(f"Energy trace is empty: {trace_csv}")

    trace = trace.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in REQUIRED_COLUMNS:
        trace[column] = pd.to_numeric(trace[column], errors="raise")

    if not trace["energy"].map(pd.notna).all():
        raise ValueError(f"Energy trace contains NaN energies: {trace_csv}")
    if not trace["temperature"].map(pd.notna).all():
        raise ValueError(f"Energy trace contains NaN temperatures: {trace_csv}")
    if (trace["n_selected"] < 0).any():
        raise ValueError(f"Energy trace contains negative selected-spin counts: {trace_csv}")
    if not trace["step"].is_monotonic_increasing:
        raise ValueError(f"Energy trace steps are not monotonic: {trace_csv}")
    if trace["step"].duplicated().any():
        raise ValueError(f"Energy trace contains duplicate steps: {trace_csv}")
    if (trace["temperature"].diff().dropna() > 1e-12).any():
        raise ValueError(f"Energy trace temperatures increase unexpectedly: {trace_csv}")

    return trace


def plot_hamiltonian_trace(
    trace_csv: Path,
    out_path: Optional[Path] = None,
    dpi: int = 180,
) -> plt.Figure:
    trace = load_energy_trace(trace_csv)

    fig, (ax_step, ax_temp) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.4),
        facecolor="#0e1117",
    )

    for ax in (ax_step, ax_temp):
        ax.set_facecolor("#0e1117")
        ax.grid(True, color="#2a2f3a", alpha=0.55, linewidth=0.8)
        ax.tick_params(colors="#c7d0d9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#39414f")

    ax_step.plot(
        trace["step"],
        trace["energy"],
        color="#4ecdc4",
        linewidth=1.8,
        marker="o",
        markersize=3.2,
    )
    ax_step.set_title("Hamiltonian vs. annealing step", color="white", fontsize=11, pad=10)
    ax_step.set_xlabel("Annealing step", color="white")
    ax_step.set_ylabel("Hamiltonian", color="white")

    ax_temp.plot(
        trace["temperature"],
        trace["energy"],
        color="#ff9f1c",
        linewidth=1.8,
        marker="o",
        markersize=3.2,
    )
    ax_temp.set_title("Hamiltonian vs. temperature", color="white", fontsize=11, pad=10)
    ax_temp.set_xlabel("Temperature", color="white")
    ax_temp.set_ylabel("Hamiltonian", color="white")
    ax_temp.invert_xaxis()

    best_idx = trace["energy"].idxmin()
    best_row = trace.loc[best_idx]
    annotation = (
        f"min H = {best_row['energy']:.3f}\n"
        f"step = {int(best_row['step'])}, T = {best_row['temperature']:.3f}\n"
        f"selected = {int(best_row['n_selected'])}"
    )
    ax_step.scatter(
        [best_row["step"]],
        [best_row["energy"]],
        s=46,
        color="#f94144",
        zorder=4,
    )
    ax_step.annotate(
        annotation,
        xy=(best_row["step"], best_row["energy"]),
        xytext=(10, 12),
        textcoords="offset points",
        color="white",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#161b22", "ec": "#49556a", "alpha": 0.95},
    )

    fig.suptitle(
        "Annealing Hamiltonian Trace",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
        print(f"  saved  {out_path}")

    return fig
