import numpy as np
import pandas as pd
from pathlib import Path

from .hits import Detector

class DataConfig:
    # particles parameters
    n_particles             = 3
    traj_radius_low         = 4
    traj_radius_high        = 20
    # detector parameters
    detector_layers         = [1, 2, 3]
    # Resolution noise (gaussian)
    sigma_res               = 0    # 0.05
    # Backgorund noise (fake hits)
    mean_fakes_per_layer    = 0    # 2


def main():
    cfg = DataConfig

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
