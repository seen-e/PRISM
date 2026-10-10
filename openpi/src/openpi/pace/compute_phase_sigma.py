"""Phase 2 (CAS) target precomputation -- proposal.md Sec. 19.

Computes, for a tianji LeRobot v3 dataset, the phase-conditional per-dimension
action std sigma_j(phase) using the gripper-closure phase segmentation from
`phase_labeling.py` (zero manual annotation, reuses the exact method already
validated by the D2 diagnostic in Sec. 18.5). Outputs are used for:

  1. sanity check of the segmentation itself (sigma_j(insert) should be
     smaller than sigma_j(pick)/sigma_j(handoff) on the fine dimensions --
     this matches the D2 result: mean joint sigma ratio insert/pick ~= 0.70);
  2. the CAS warm-up regression target (`Pi0.compute_cas_warmup_loss`): each
     training frame's target sigma is looked up via (episode, frame) -> phase
     -> sigma_j(phase), see `phase_target.py`;
  3. the initial bias of ScaleHead's last layer (global sigma, phases pooled)
     so CAS starts close to a phase-agnostic estimate before warm-up.

Only reads `observation.state` (for gripper channels) and `action` (for
sigma statistics) columns from the raw parquet -- no video decoding, so this
is fast even over the full dataset (same trick as the D2 script).

Usage:
    python -m openpi.pace.compute_phase_sigma \\
        --repo-id local/tianji_joint_cmd_645newop_dagger_20260625_stride2 \\
        --out-dir assets/pace/cas
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq
import tyro

from openpi.pace.phase_labeling import PHASES, label_phases
from openpi.shared import normalize as _normalize


@dataclasses.dataclass
class Args:
    repo_id: str
    root: pathlib.Path | None = None
    out_dir: pathlib.Path = pathlib.Path("assets/pace/cas")
    # Tianji joint-cmd 16-dim action layout: 7 DoF x 2 arms + 2 grippers,
    # grippers at the last two indices. Override if your layout differs.
    gripper_left_idx: int = 14
    gripper_right_idx: int = 15
    smooth_window: int = 7
    sustain_frames: int = 5
    state_key: str = "observation.state"
    action_key: str = "action"


def _episode_table_paths(root: pathlib.Path, info: dict) -> list[pathlib.Path]:
    data_path_tmpl = info["data_path"]
    total_chunks = info.get("total_chunks", 1)
    total_files = info.get("total_files", 1)
    paths = []
    for chunk_index in range(total_chunks):
        for file_index in range(total_files):
            p = root / data_path_tmpl.format(chunk_index=chunk_index, file_index=file_index)
            if p.exists():
                paths.append(p)
    if not paths:
        # Some exports only have a single chunk-000/file-000.parquet regardless of totals.
        p = root / data_path_tmpl.format(chunk_index=0, file_index=0)
        if p.exists():
            paths.append(p)
    return paths


def main(args: Args) -> None:
    root = args.root or (pathlib.Path.home() / ".cache" / "huggingface" / "lerobot" / args.repo_id)
    with (root / "meta" / "info.json").open() as f:
        info = json.load(f)

    table_paths = _episode_table_paths(root, info)
    if not table_paths:
        raise FileNotFoundError(f"No parquet data files found under {root}")

    stats = {phase: _normalize.RunningStats() for phase in PHASES}
    boundaries: dict[int, dict[str, int]] = {}
    n_episodes = 0

    for path in table_paths:
        table = pq.read_table(path, columns=[args.state_key, args.action_key, "episode_index"])
        episode_index_col = np.asarray(table["episode_index"].to_pylist())
        state = np.stack(table[args.state_key].to_pylist())
        action = np.stack(table[args.action_key].to_pylist())
        # `action` may be a per-frame chunk (T, action_horizon, D); use the
        # first step (the actually-executed command) for sigma statistics.
        if action.ndim == 3:
            action = action[:, 0, :]

        for ep in np.unique(episode_index_col):
            mask = episode_index_col == ep
            ep_state = state[mask]
            ep_action = action[mask]

            gl = ep_state[:, args.gripper_left_idx]
            gr = ep_state[:, args.gripper_right_idx]
            labels = label_phases(gl, gr, smooth_window=args.smooth_window, sustain_frames=args.sustain_frames)

            for phase in PHASES:
                phase_mask = labels == phase
                if phase_mask.any():
                    stats[phase].update(ep_action[phase_mask])

            pick_end = int(np.argmax(labels != "pick")) if np.any(labels != "pick") else len(labels)
            rest = labels[pick_end:]
            handoff_end = pick_end + (int(np.argmax(rest != "handoff")) if np.any(rest != "handoff") else len(rest))
            boundaries[int(ep)] = {
                "pick_end": pick_end,
                "handoff_end": handoff_end,
                "length": int(len(labels)),
            }
            n_episodes += 1

    phase_sigma = {phase: stats[phase].get_statistics().std.tolist() for phase in PHASES}
    global_sigma_init = np.mean([phase_sigma[p] for p in PHASES], axis=0).tolist()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "phase_sigma.json").write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "n_episodes": n_episodes,
                "gripper_left_idx": args.gripper_left_idx,
                "gripper_right_idx": args.gripper_right_idx,
                "phase_sigma": phase_sigma,
                "global_sigma_init": global_sigma_init,
            },
            indent=2,
        )
    )
    (args.out_dir / "phase_boundaries.json").write_text(json.dumps(boundaries, indent=2))

    print(f"[compute_phase_sigma] {n_episodes} episodes processed from {len(table_paths)} parquet file(s).")
    for phase in PHASES:
        print(f"  sigma[{phase}] (first 6 dims) = {np.round(phase_sigma[phase][:6], 4).tolist()}")
    ratio = np.mean(phase_sigma["insert"]) / np.mean(phase_sigma["pick"])
    print(f"  sanity check: mean(sigma[insert]) / mean(sigma[pick]) = {ratio:.3f}  (expect < 1)")
    print(f"Wrote: {args.out_dir / 'phase_sigma.json'}")
    print(f"Wrote: {args.out_dir / 'phase_boundaries.json'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
