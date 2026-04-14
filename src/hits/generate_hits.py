import numpy as np
import pandas as pd
from typing import Tuple, List

class Trajectory:
    """ Class representing one particle trajectory """
    def __init__(
            self,
            trajectory_id: int, 
            phi: float, 
            radius: float, 
            curvature: int,
            ):
        self.trajectory_id = trajectory_id
        self.phi = phi
        self.radius = radius
        self.curvature = curvature
    
    def compute_intersection(
            self,
            detector_radius: float,
            ) -> Tuple[float, float]:
        """ Computes the hit point on one detector 
         Assuming the detector is circular with radius detector_radius 
         returns the coordinates of the hit """
        
        if self.radius <= 0:
            raise ValueError(f"Invalid trajectory radius: {self.radius} (must be > 0)")
        if detector_radius <= 0:
            raise ValueError(f"Invalid detector radius: {detector_radius} (must be > 0)")
        if detector_radius > 2 * self.radius:
            raise ValueError(
                f"Trajectory {self.trajectory_id} does not intersect detector: "
                f"detector_radius={detector_radius} > 2*radius={2*self.radius}"
            )

        alpha = np.arcsin(detector_radius / (2 * self.radius))
        hit_angle = self.phi + self.curvature * alpha

        hit_x = detector_radius * np.cos(hit_angle)
        hit_y = detector_radius * np.sin(hit_angle)
        return (hit_x, hit_y)
    
    
class Detector:
    def __init__(
            self,
            detector_radii: List,
            n_particles: int, 
            traj_radius_low: float,
            traj_radius_high: float
            ):
        if len(detector_radii) == 0:
            raise ValueError("detector_radii must contain at least one layer radius")
        if any(radius <= 0 for radius in detector_radii):
            raise ValueError("All detector_radii must be strictly positive")
        if n_particles <= 0:
            raise ValueError(f"n_particles must be > 0, got {n_particles}")
        if traj_radius_low <= 0 or traj_radius_high <= 0:
            raise ValueError(
                "Trajectory radius bounds must be strictly positive "
                f"(got low={traj_radius_low}, high={traj_radius_high})"
            )
        if traj_radius_low > traj_radius_high:
            raise ValueError(
                f"Invalid trajectory radius bounds: low={traj_radius_low} > high={traj_radius_high}"
            )

        # Compatibility check: every generated trajectory must intersect every layer.
        # Since r is sampled in [traj_radius_low, traj_radius_high], the worst case is traj_radius_low.
        max_layer_radius = max(detector_radii)
        min_required_traj_radius = max_layer_radius / 2.0
        if traj_radius_low < min_required_traj_radius:
            raise ValueError(
                "Incompatible trajectory radius bounds for this detector: "
                f"traj_radius_low={traj_radius_low} is too small for max detector radius "
                f"{max_layer_radius}. Need traj_radius_low >= {min_required_traj_radius} "
                "to guarantee intersections on all layers."
            )

        self.R = detector_radii
        self.n = n_particles
        # generate particles in the experiment with random trajectories
        phi = np.random.uniform(0, 2*np.pi, self.n)
        r = np.random.uniform(traj_radius_low, traj_radius_high, self.n)
        k = 2*np.random.randint(0, 2, self.n) - 1
        self.trajectories = [Trajectory(i, phi[i], r[i], k[i]) for i in range(self.n)]

    def get_hits(self) -> pd.DataFrame:
        """Return one row per hit with track and detector layer metadata."""
        rows = []
        for layer_id, layer_radius in enumerate(self.R):
            for trajectory in self.trajectories:
                hit_x, hit_y = trajectory.compute_intersection(layer_radius)
                rows.append(
                    {
                        "track_id": trajectory.trajectory_id,
                        "layer_id": layer_id,
                        "layer_radius": float(layer_radius),
                        "hit_x": float(hit_x),
                        "hit_y": float(hit_y),
                        "hit_phi": float(np.arctan2(hit_y, hit_x)),
                    }
                )

        return (
            pd.DataFrame(rows)
            .sort_values(["layer_id", "track_id"])
            .reset_index(drop=True)
        )

