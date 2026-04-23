import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field

from .hits import Detector

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'pyyaml'. Install dependencies first (e.g. `uv sync`)."
    ) from exc


@dataclass(frozen=True)
class DataConfig:
    # particles parameters
    n_particles: int = 20
    traj_radius_low: float = 4
    traj_radius_high: float = 20
    # detector parameters
    detector_layers: list[float] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    # Resolution noise (gaussian)
    sigma_res: float = 0.0  # 0.05
    # Backgorund noise (fake hits)
    mean_fakes_per_layer: float = 0.0  # 2

    @classmethod
    def from_yaml(cls, config_path: Path | None = None) -> "DataConfig":
        cfg = cls()
        root = Path(__file__).resolve().parents[2]
        cfg_path = config_path or (root / "scripts/config.yaml")
        if not cfg_path.exists():
            return cfg

        with cfg_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}

        if not isinstance(loaded, dict):
            raise ValueError(f"Config root must be a mapping in {cfg_path}")

        generation = loaded.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("generation section must be a mapping")

        data_cfg = generation.get("data", {})
        if not isinstance(data_cfg, dict):
            raise ValueError("generation.data must be a mapping")

        detector_layers = data_cfg.get("detector_layers", cfg.detector_layers)
        if not isinstance(detector_layers, list) or len(detector_layers) == 0:
            raise ValueError("generation.data.detector_layers must be a non-empty list")

        return cls(
            n_particles=int(data_cfg.get("n_particles", cfg.n_particles)),
            traj_radius_low=float(data_cfg.get("traj_radius_low", cfg.traj_radius_low)),
            traj_radius_high=float(data_cfg.get("traj_radius_high", cfg.traj_radius_high)),
            detector_layers=[float(r) for r in detector_layers],
            sigma_res=float(data_cfg.get("sigma_res", cfg.sigma_res)),
            mean_fakes_per_layer=float(
                data_cfg.get("mean_fakes_per_layer", cfg.mean_fakes_per_layer)
            ),
        )


def main():
    cfg = DataConfig.from_yaml()

    print("--- Generating hits ---")
    print(f" #particles={cfg.n_particles}")
    print(f" detector_layers={cfg.detector_layers}")

    experiment = Detector(cfg.detector_layers, cfg.n_particles, cfg.traj_radius_low, cfg.traj_radius_high)
    clean_hits = experiment.get_hits()
    n_real_hits = len(clean_hits)

    # add position smearing
    noisy_hits = clean_hits.copy()
    noisy_hits["hit_x"] = noisy_hits["hit_x"] + np.random.normal(
        loc=0.0, scale=cfg.sigma_res, size=len(noisy_hits)
    )
    noisy_hits["hit_y"] = noisy_hits["hit_y"] + np.random.normal(
        loc=0.0, scale=cfg.sigma_res, size=len(noisy_hits)
    )
    noisy_hits.insert(0, "hit_id", np.arange(n_real_hits, dtype=int))

    # generate fake hits due to noise
    fake_rows = []
    for layer_id, layer_radius in enumerate(cfg.detector_layers):
        n_fake = np.random.poisson(lam=cfg.mean_fakes_per_layer)
        angles = np.random.uniform(0, 2 * np.pi, n_fake)
        x_fake = layer_radius * np.cos(angles)
        y_fake = layer_radius * np.sin(angles)

        for i in range(n_fake):
            fake_rows.append(
                {
                    "track_id": -1,
                    "layer_id": layer_id,
                    "layer_radius": float(layer_radius),
                    "hit_x": float(x_fake[i]),
                    "hit_y": float(y_fake[i]),
                    "hit_phi": float(angles[i]),
                }
            )

    fake_hits = pd.DataFrame(fake_rows, columns=clean_hits.columns)
    fake_hits.insert(
        0,
        "hit_id",
        np.arange(n_real_hits, n_real_hits + len(fake_hits), dtype=int),
    )
    all_hits = pd.concat([noisy_hits, fake_hits], ignore_index=True)
    all_hits = all_hits.sort_values(["layer_id", "track_id"]).reset_index(drop=True)

    # Save the files
    data_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_path = data_dir / "ground_truth_hits.csv"
    training_path = data_dir / "training_hits.csv"
    ground_truth_hits = clean_hits.copy()
    ground_truth_hits.insert(0, "hit_id", np.arange(n_real_hits, dtype=int))
    ground_truth_hits.to_csv(ground_truth_path, index=False)
    training_hits = all_hits.drop(columns=["track_id"]).copy()
    training_hits.to_csv(training_path, index=False)

    print(" hits saved!")


if __name__ == "__main__":
    main()
