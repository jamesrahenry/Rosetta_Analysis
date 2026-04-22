"""
viz_coords.py — Shared coordinate loader for all visualization scripts.

Loads the precomputed PCA coordinates from shared_coords.npz so every
viz uses the same spatial frame regardless of which features are displayed.

Usage:
    from viz_coords import load_shared_coords

    coords = load_shared_coords("EleutherAI/pythia-1.4b")
    x, y, z = coords.get("concept", "credibility", layer=5)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


class SharedCoords:
    """Precomputed PCA coordinate frame for one model."""

    def __init__(self, npz_path: Path):
        data = np.load(npz_path, allow_pickle=False)
        self.coords_2d = data["coords_2d"]
        self.label_types = data["label_types"]
        self.label_ids = data["label_ids"]
        self.label_layers = data["label_layers"]
        self.axis_ranges = data["axis_ranges"]
        self.n_layers = int(data["n_layers"][0])
        self.explained_variance = data["explained_variance_ratio"]

        # Build index for fast lookup
        self._index: dict[tuple[str, str, int], int] = {}
        for i in range(len(self.label_types)):
            key = (str(self.label_types[i]),
                   str(self.label_ids[i]),
                   int(self.label_layers[i]))
            self._index[key] = i

    def get(self, label_type: str, label_id: str, layer: int):
        """Get (x, y, z) for a vector, or (None, None, None) if missing."""
        key = (label_type, str(label_id), layer)
        idx = self._index.get(key)
        if idx is None:
            return None, None, None
        x = float(self.coords_2d[idx, 0])
        y = float(self.coords_2d[idx, 1])
        z = 100.0 * layer / self.n_layers
        return x, y, z

    @property
    def x_range(self) -> tuple[float, float]:
        return float(self.axis_ranges[0]), float(self.axis_ranges[1])

    @property
    def y_range(self) -> tuple[float, float]:
        return float(self.axis_ranges[2]), float(self.axis_ranges[3])


def load_shared_coords(model_id: str) -> SharedCoords:
    """Load shared coordinates for a model, or raise if not computed."""
    model_slug = model_id.replace("/", "_").replace("-", "_")
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if not d.name.startswith(f"deepdive_{model_slug}"):
            continue
        npz = d / "shared_coords.npz"
        if npz.exists():
            log.info("Loading shared coords from %s", npz)
            return SharedCoords(npz)

    raise FileNotFoundError(
        f"No shared_coords.npz for {model_id}. "
        f"Run: python src/compute_shared_pca.py --model {model_id}"
    )
