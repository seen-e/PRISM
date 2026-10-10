"""Runtime lookup: (episode, frame) -> CAS warm-up target sigma_j(phase).

Loads the `phase_sigma.json` / `phase_boundaries.json` produced by
`compute_phase_sigma.py` and exposes a vectorized lookup. Intended to be
called from a dataset transform that adds a `cas_sigma_target` field next to
`action` for each training sample, so `Pi0.compute_cas_warmup_loss` can be
plugged into the existing data pipeline without touching the core
flow-matching batch format.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from openpi.pace.phase_labeling import PHASES


class PhaseSigmaTargets:
    def __init__(self, cas_asset_dir: pathlib.Path | str):
        cas_asset_dir = pathlib.Path(cas_asset_dir)
        doc = json.loads((cas_asset_dir / "phase_sigma.json").read_text())
        self.phase_sigma = {p: np.asarray(doc["phase_sigma"][p], dtype=np.float32) for p in PHASES}
        self.global_sigma_init = np.asarray(doc["global_sigma_init"], dtype=np.float32)
        self.boundaries = {
            int(ep): v for ep, v in json.loads((cas_asset_dir / "phase_boundaries.json").read_text()).items()
        }

    def phase_of(self, episode_index: int, frame_index: int) -> str:
        b = self.boundaries[int(episode_index)]
        if frame_index < b["pick_end"]:
            return "pick"
        if frame_index < b["handoff_end"]:
            return "handoff"
        return "insert"

    def sigma_target(self, episode_index: int, frame_index: int) -> np.ndarray:
        return self.phase_sigma[self.phase_of(episode_index, frame_index)]

    def sigma_target_batch(self, episode_indices: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
        return np.stack(
            [self.sigma_target(ep, t) for ep, t in zip(episode_indices, frame_indices, strict=True)]
        ).astype(np.float32)
