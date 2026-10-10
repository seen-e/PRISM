"""Small dataset adapter for attaching precomputed PPS targets."""

from __future__ import annotations

from collections.abc import Sequence
import pathlib
from typing import Any

import numpy as np


class PPSWeightedDataset:
    """Attach one scalar ``pps_weight`` to every item in a map-style dataset.

    The order of ``weights`` must exactly match the wrapped dataset. The wrapper
    copies dictionary samples before inserting the target, so the source dataset
    is not mutated.
    """

    def __init__(self, dataset: Any, weights: str | pathlib.Path | Sequence[float] | np.ndarray):
        self.dataset = dataset
        self.weights = np.load(weights) if isinstance(weights, (str, pathlib.Path)) else np.asarray(weights)
        self.weights = np.asarray(self.weights, dtype=np.float32).reshape(-1)
        if len(self.weights) != len(dataset):
            raise ValueError(f"PPS length {len(self.weights)} does not match dataset length {len(dataset)}")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("PPS weights must be finite")
        if np.any((self.weights < 0.0) | (self.weights > 1.0)):
            raise ValueError("PPS weights must lie in [0, 1]")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        if not isinstance(sample, dict):
            raise TypeError("PPSWeightedDataset expects dictionary samples")
        output = dict(sample)
        output["pps_weight"] = self.weights[index]
        return output


class EpisodePPSLookup:
    """Lookup PPS by ``(dataset_name, trajectory_id, frame_index)``.

    ``root`` contains one ``<dataset_name>.npz`` file per dataset. Each file
    stores parallel one-dimensional arrays named ``trajectory_id``,
    ``frame_index``, and ``pps_weight``. This identity-based format stays
    correct when a mixture dataset samples frames randomly.
    """

    def __init__(self, root: str | pathlib.Path):
        self.root = pathlib.Path(root)
        self._cache: dict[str, dict[tuple[int, int], np.float32]] = {}

    def _load(self, dataset_name: str) -> dict[tuple[int, int], np.float32]:
        if dataset_name in self._cache:
            return self._cache[dataset_name]
        path = self.root / f"{dataset_name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing PPS lookup file: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {"trajectory_id", "frame_index", "pps_weight"}
            if not required.issubset(archive.files):
                raise ValueError(f"{path} must contain arrays {sorted(required)}")
            trajectory = np.asarray(archive["trajectory_id"]).reshape(-1)
            frame = np.asarray(archive["frame_index"]).reshape(-1)
            weight = np.asarray(archive["pps_weight"], dtype=np.float32).reshape(-1)
        if not (len(trajectory) == len(frame) == len(weight)):
            raise ValueError(f"Mismatched PPS array lengths in {path}")
        if not np.all(np.isfinite(weight)) or np.any((weight < 0.0) | (weight > 1.0)):
            raise ValueError(f"Invalid PPS values in {path}")
        lookup = {(int(ep), int(step)): rho for ep, step, rho in zip(trajectory, frame, weight, strict=True)}
        if len(lookup) != len(weight):
            raise ValueError(f"Duplicate (trajectory_id, frame_index) entries in {path}")
        self._cache[dataset_name] = lookup
        return lookup

    def __call__(self, dataset_name: str, trajectory_id: int, frame_index: int) -> np.float32:
        key = (int(trajectory_id), int(frame_index))
        try:
            return self._load(dataset_name)[key]
        except KeyError as exc:
            raise KeyError(f"No PPS target for {dataset_name} trajectory={key[0]} frame={key[1]}") from exc
