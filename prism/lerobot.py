"""Generate PRISM PPS targets from a local LeRobot dataset."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from prism.pps import PPSConfig, compute_episode_pps


def compute_pps_records(
    state: np.ndarray,
    episode_id: np.ndarray,
    frame_index: np.ndarray,
    *,
    global_index: np.ndarray | None = None,
    config: PPSConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PPS and return arrays in deterministic dataset order.

    PPS is computed episode-by-episode in frame order. Returned arrays are
    sorted by ``global_index`` when supplied, otherwise by episode/frame.
    """

    state = np.asarray(state)
    episode_id = np.asarray(episode_id).reshape(-1)
    frame_index = np.asarray(frame_index).reshape(-1)
    count = len(state)
    if len(episode_id) != count or len(frame_index) != count:
        raise ValueError("state, episode_id, and frame_index must have equal lengths")
    if global_index is not None and len(np.asarray(global_index).reshape(-1)) != count:
        raise ValueError("global_index must have the same length as state")

    rho = np.empty(count, dtype=np.float32)
    for episode in np.unique(episode_id):
        indices = np.flatnonzero(episode_id == episode)
        episode_order = indices[np.argsort(frame_index[indices], kind="stable")]
        episode_frames = frame_index[episode_order]
        if len(np.unique(episode_frames)) != len(episode_frames):
            raise ValueError(f"duplicate frame_index values in episode {episode}")
        rho[episode_order] = compute_episode_pps(state[episode_order], config)

    if global_index is None:
        order = np.lexsort((frame_index, episode_id))
    else:
        global_index = np.asarray(global_index).reshape(-1)
        if len(np.unique(global_index)) != count:
            raise ValueError("global_index values must be unique")
        order = np.argsort(global_index, kind="stable")
    return rho[order], episode_id[order], frame_index[order]


def _stack_vector_column(values: Sequence, name: str) -> np.ndarray:
    try:
        array = np.stack([np.asarray(value, dtype=np.float64) for value in values])
    except ValueError as exc:
        raise ValueError(f"Parquet column {name!r} does not contain fixed-size vectors") from exc
    if array.ndim != 2:
        raise ValueError(f"Parquet column {name!r} must produce [N, D], got {array.shape}")
    return array


def load_lerobot_records(
    dataset_root: str | Path,
    *,
    state_key: str | None,
    joint_key: str | None,
    eef_key: str | None,
    gripper_key: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Read state and frame identities from LeRobot v2/v3 parquet files."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Install the data extra first: pip install -e '.[data]'") from exc

    root = Path(dataset_root)
    parquet_paths = sorted((root / "data").glob("**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    if state_key is None and not all((joint_key, eef_key, gripper_key)):
        raise ValueError("Provide --state-key or all of --joint-key/--eef-key/--gripper-key")

    state_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    global_parts: list[np.ndarray] = []
    has_global_index = True

    for path in parquet_paths:
        schema_names = set(pq.read_schema(path).names)
        episode_column = "episode_index" if "episode_index" in schema_names else "episode_id"
        frame_column = "frame_index" if "frame_index" in schema_names else "index"
        required = [episode_column, frame_column]
        vector_keys = [state_key] if state_key is not None else [joint_key, eef_key, gripper_key]
        required.extend(key for key in vector_keys if key is not None)
        missing = [name for name in required if name not in schema_names]
        if missing:
            raise KeyError(f"{path} is missing columns: {missing}")
        columns = list(dict.fromkeys(required + (["index"] if "index" in schema_names else [])))
        table = pq.read_table(path, columns=columns)

        vectors = [_stack_vector_column(table[key].to_pylist(), key) for key in vector_keys if key is not None]
        state_parts.append(vectors[0] if state_key is not None else np.concatenate(vectors, axis=1))
        episode_parts.append(np.asarray(table[episode_column].to_numpy()))
        frame_parts.append(np.asarray(table[frame_column].to_numpy()))
        if "index" in table.column_names:
            global_parts.append(np.asarray(table["index"].to_numpy()))
        else:
            has_global_index = False

    state = np.concatenate(state_parts, axis=0)
    episode = np.concatenate(episode_parts, axis=0)
    frame = np.concatenate(frame_parts, axis=0)
    global_index = np.concatenate(global_parts, axis=0) if has_global_index else None
    return state, episode, frame, global_index


def write_pps_outputs(
    output_dir: str | Path,
    dataset_name: str,
    rho: np.ndarray,
    episode_id: np.ndarray,
    frame_index: np.ndarray,
    config: PPSConfig,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    flat_path = output / "pps_weights.npy"
    lookup_path = output / f"{dataset_name}.npz"
    np.save(flat_path, np.asarray(rho, dtype=np.float32))
    np.savez_compressed(
        lookup_path,
        trajectory_id=np.asarray(episode_id),
        frame_index=np.asarray(frame_index),
        pps_weight=np.asarray(rho, dtype=np.float32),
    )
    metadata = {
        "dataset_name": dataset_name,
        "num_frames": int(len(rho)),
        "num_episodes": int(len(np.unique(episode_id))),
        "pps_config": dataclasses.asdict(config),
        "openpi_output": flat_path.name,
        "starvla_output": lookup_path.name,
    }
    (output / "pps_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return flat_path, lookup_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-name", default=None, help="Defaults to the dataset directory name")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--joint-key", default=None)
    parser.add_argument("--eef-key", default=None)
    parser.add_argument("--gripper-key", default=None)
    parser.add_argument("--joint-dims", type=int, default=14)
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--pause-percentile", type=float, default=5.0)
    parser.add_argument("--pause-min", type=float, default=1e-4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    state_key = args.state_key
    if any((args.joint_key, args.eef_key, args.gripper_key)):
        state_key = None
    state, episode, frame, global_index = load_lerobot_records(
        args.dataset_root,
        state_key=state_key,
        joint_key=args.joint_key,
        eef_key=args.eef_key,
        gripper_key=args.gripper_key,
    )
    config = PPSConfig(
        joint_dims=args.joint_dims,
        smooth_window=args.smooth_window,
        pause_percentile=args.pause_percentile,
        pause_min=args.pause_min,
    )
    rho, episode, frame = compute_pps_records(
        state,
        episode,
        frame,
        global_index=global_index,
        config=config,
    )
    flat_path, lookup_path = write_pps_outputs(
        args.output_dir,
        args.dataset_name or args.dataset_root.name,
        rho,
        episode,
        frame,
        config,
    )
    print(f"Wrote OpenPI targets: {flat_path}")
    print(f"Wrote StarVLA targets: {lookup_path}")


if __name__ == "__main__":
    main()
