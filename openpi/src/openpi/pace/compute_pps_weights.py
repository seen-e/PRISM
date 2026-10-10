"""PPS (Precision Proxy from Speed) weight computation.

Speed modes:
  - state_joint_delta (v1): slow joint motion -> high rho (flags pauses)
  - v2a: mask pauses; rank slowness only among active-motion frames
  - v2b: v2a-style pause mask + EEF slowness + gripper activity fusion
  - v3_phase: rule-based transport / approach / manipulate / pause phases (discrete rho)
  - v4_wire / universal: continuous rho — grasp engagement × slowness among active frames
  - action_l2 (legacy): ||action||_2, not recommended
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import tyro

SpeedMode = Literal["state_joint_delta", "v2a", "v2b", "v3_phase", "v4_wire", "universal", "action_l2"]


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    kernel = np.ones(window) / window
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


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
        p = root / data_path_tmpl.format(chunk_index=0, file_index=0)
        if p.exists():
            paths.append(p)
    return paths


def _joint_speed(state: np.ndarray, joint_dims: int) -> np.ndarray:
    joints = state[:, :joint_dims]
    if joints.shape[0] < 2:
        return np.zeros(joints.shape[0], dtype=np.float64)
    delta = np.linalg.norm(np.diff(joints, axis=0), axis=1)
    return np.concatenate([[delta[0]], delta]).astype(np.float64)


def _eef_speed(state: np.ndarray) -> np.ndarray:
    """Max end-effector displacement per frame across both arms (xyz)."""
    eef = state[:, 14:20].reshape(-1, 2, 3)
    if eef.shape[0] < 2:
        return np.zeros(eef.shape[0], dtype=np.float64)
    per_arm = np.linalg.norm(np.diff(eef, axis=0), axis=2)
    delta = per_arm.max(axis=1)
    return np.concatenate([[delta[0]], delta]).astype(np.float64)


def _gripper_activity(state: np.ndarray) -> np.ndarray:
    grip = state[:, 20:22]
    if grip.shape[0] < 2:
        return np.zeros(grip.shape[0], dtype=np.float64)
    dg = np.linalg.norm(np.diff(grip, axis=0), axis=1)
    dg = np.concatenate([[dg[0]], dg])
    dg_score = dg / (dg.max() + 1e-8)
    engaged = grip.max(axis=1)
    engaged_score = engaged / (engaged.max() + 1e-8)
    return 0.5 * dg_score + 0.5 * engaged_score


def _sigmoid_slow_score(values: np.ndarray, active: np.ndarray, eps: float) -> np.ndarray:
    """Higher when value is lower than active-frame mean (slow among active)."""
    out = np.zeros_like(values, dtype=np.float64)
    if not np.any(active):
        return out
    v = values[active]
    v_bar = v.mean()
    s_v = v.std() + eps
    out[active] = 1.0 / (1.0 + np.exp(-(v_bar - v) / s_v))
    return out


def _pause_mask(dq: np.ndarray, pause_percentile: float, pause_min: float) -> np.ndarray:
    thresh = float(np.percentile(dq, pause_percentile))
    thresh = max(thresh, pause_min)
    return dq >= thresh


def _normalize_episode(rho: np.ndarray, eps: float) -> np.ndarray:
    rho_min, rho_max = rho.min(), rho.max()
    return (rho - rho_min) / (rho_max - rho_min + eps)


def _gripper_signals(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grip = state[:, 20:22]
    grip_max = grip.max(axis=1)
    if grip.shape[0] < 2:
        return grip_max, np.zeros_like(grip_max)
    dg = np.linalg.norm(np.diff(grip, axis=0), axis=1)
    dg = np.concatenate([[dg[0]], dg])
    return grip_max, dg


def _percentile_rank(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map masked values to [0, 1] by within-mask rank (continuous, no hard bins)."""
    out = np.zeros(values.shape[0], dtype=np.float64)
    if not np.any(mask):
        return out
    v = values[mask]
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(v))
    out[mask] = ranks
    return out


def _universal_rho(
    state: np.ndarray,
    *,
    joint_dims: int,
    eps: float,
    pause_percentile: float,
    pause_min: float,
    universal_pause_rho: float = 0.5,
    universal_open_baseline: float = 0.35,
    universal_speed_dq_weight: float = 0.5,
    universal_speed_deef_weight: float = 0.5,
    universal_absolute_speed_ceiling: float = 0.0,
    universal_absolute_transition_fraction: float = 0.4,
    universal_grasp_acquire_penalty: float = 0.70,
    universal_grasp_settle_window: int = 11,
    universal_grasp_trust_floor: float = 0.25,
    universal_grip_quiescent_penalty: float = 0.75,
    universal_grip_activity_window: int = 13,
) -> np.ndarray:
    """Universal PPS: slow active motion while grasping -> high rho (continuous in [0, 1]).

    Applies to any grasp-and-move task (pick-place, insertion, handover, wiping, …):
      - Quality demos move slower when placement precision matters.
      - rho rises with slowness *among active (non-pause) frames in the episode*.
      - Grip engagement scales precision; open-grip motion stays near open_baseline.
      - Grasp acquisition (gripper recently closing) suppresses precision credit:
        slow motion while closing is "grasping", not fine placement.
      - Grip activity (handover / finger adjustment with static arms) suppresses credit:
        slowness counts only when the gripper has been quiescent recently.
      - Pause frames are neutral (not rewarded).
    """
    dq = _joint_speed(state, joint_dims)
    deef = _eef_speed(state)
    grip_max, dg = _gripper_signals(state)
    active = _pause_mask(dq, pause_percentile, pause_min)

    rho = np.full(state.shape[0], universal_pause_rho, dtype=np.float64)
    if not np.any(active):
        return rho

    g_lo = float(grip_max.min())
    g_hi = float(grip_max.max())
    grip_eng = np.clip((grip_max - g_lo) / (g_hi - g_lo + eps), 0.0, 1.0)

    w_sum = universal_speed_dq_weight + universal_speed_deef_weight + eps
    speed = (universal_speed_dq_weight * dq + universal_speed_deef_weight * deef) / w_sum
    slowness = 1.0 - _percentile_rank(speed, active)

    # Episode ranks alone always label some frames as slow. This absolute gate
    # prevents ordinary transport from becoming high-precision solely by rank.
    if universal_absolute_speed_ceiling > 0.0:
        transition = max(
            universal_absolute_speed_ceiling * universal_absolute_transition_fraction,
            eps,
        )
        absolute_slow = np.clip(
            (universal_absolute_speed_ceiling - speed) / transition,
            0.0,
            1.0,
        )
    else:
        absolute_slow = np.ones_like(speed)

    # Grasp-acquisition gate: recently closing gripper -> do not treat slowness as precision.
    d_grip_eng = np.maximum(np.diff(grip_eng, prepend=grip_eng[0]), 0.0)
    if universal_grasp_settle_window > 1:
        kernel = np.ones(universal_grasp_settle_window) / universal_grasp_settle_window
        recent_close = np.convolve(d_grip_eng, kernel, mode="full")[: state.shape[0]]
    else:
        recent_close = d_grip_eng
    close_scale = float(np.percentile(recent_close[active], 92)) + eps
    grasp_acquire = np.clip(recent_close / close_scale, 0.0, 1.0)
    trust_acquire = np.clip(
        1.0 - universal_grasp_acquire_penalty * grasp_acquire,
        universal_grasp_trust_floor,
        1.0,
    )

    # Grip-activity gate: arm static + fingers/gripper adjusting (handover) != fine placement.
    if universal_grip_activity_window > 1:
        dg_kernel = np.ones(universal_grip_activity_window) / universal_grip_activity_window
        recent_dg = np.convolve(dg, dg_kernel, mode="full")[: state.shape[0]]
    else:
        recent_dg = dg
    dg_scale = float(np.percentile(recent_dg[active], 92)) + eps
    grip_active = np.clip(recent_dg / dg_scale, 0.0, 1.0)
    trust_quiescent = np.clip(
        1.0 - universal_grip_quiescent_penalty * grip_active,
        universal_grasp_trust_floor,
        1.0,
    )
    trust = trust_acquire * trust_quiescent

    rho_active = (
        grip_eng[active] * slowness[active] * trust[active]
        + (1.0 - grip_eng[active]) * universal_open_baseline
    )
    above_neutral = rho_active > universal_pause_rho
    rho_active[above_neutral] = universal_pause_rho + (
        rho_active[above_neutral] - universal_pause_rho
    ) * absolute_slow[active][above_neutral]
    rho[active] = np.clip(rho_active, 0.0, 1.0)
    return rho


# Backward-compatible alias (v4_wire now uses the universal rule).
_wire_continuous_rho = _universal_rho


def episode_precision_labels(
    state: np.ndarray,
    rho: np.ndarray | None = None,
    *,
    joint_dims: int = 14,
    eps: float = 1e-8,
    pause_percentile: float = 5.0,
    pause_min: float = 1e-4,
    coarse_thresh: float = 0.38,
    fine_thresh: float = 0.62,
    grip_eng_thresh: float = 0.35,
    grasp_acquire_thresh: float = 0.45,
    grip_adjust_thresh: float = 0.45,
    **universal_kw,
) -> list[str]:
    """Visualization tags derived from continuous rho (not used in rho computation)."""
    if rho is None:
        rho = _universal_rho(
            state,
            joint_dims=joint_dims,
            eps=eps,
            pause_percentile=pause_percentile,
            pause_min=pause_min,
            **universal_kw,
        )
    dq = _joint_speed(state, joint_dims)
    active = _pause_mask(dq, pause_percentile, pause_min)
    grip_max, dg = _gripper_signals(state)
    g_lo = float(grip_max.min())
    g_hi = float(grip_max.max())
    grip_eng = np.clip((grip_max - g_lo) / (g_hi - g_lo + eps), 0.0, 1.0)
    d_grip_eng = np.maximum(np.diff(grip_eng, prepend=grip_eng[0]), 0.0)
    settle = universal_kw.get("universal_grasp_settle_window", 11)
    if settle > 1:
        kernel = np.ones(settle) / settle
        recent_close = np.convolve(d_grip_eng, kernel, mode="full")[: state.shape[0]]
    else:
        recent_close = d_grip_eng
    active_mask = active
    close_scale = float(np.percentile(recent_close[active_mask], 92)) + eps if np.any(active_mask) else 1.0
    grasp_acquire = np.clip(recent_close / close_scale, 0.0, 1.0)
    act_win = universal_kw.get("universal_grip_activity_window", 13)
    if act_win > 1:
        dg_kernel = np.ones(act_win) / act_win
        recent_dg = np.convolve(dg, dg_kernel, mode="full")[: state.shape[0]]
    else:
        recent_dg = dg
    dg_scale = float(np.percentile(recent_dg[active_mask], 92)) + eps if np.any(active_mask) else 1.0
    grip_active = np.clip(recent_dg / dg_scale, 0.0, 1.0)

    names: list[str] = []
    for i in range(state.shape[0]):
        if not active[i]:
            names.append("pause")
        elif grasp_acquire[i] >= grasp_acquire_thresh:
            names.append("grasp_acquire")
        elif grip_active[i] >= grip_adjust_thresh:
            names.append("grip_adjust")
        elif grip_eng[i] < grip_eng_thresh:
            names.append("free_motion")
        elif rho[i] <= coarse_thresh:
            names.append("coarse")
        elif rho[i] >= fine_thresh:
            names.append("fine")
        else:
            names.append("transition")
    return names


# Backward-compatible alias.
episode_wire_labels = episode_precision_labels


def _phase_masks(
    state: np.ndarray,
    *,
    joint_dims: int,
    eps: float,
    pause_percentile: float,
    pause_min: float,
    phase_transport_speed_pct: float,
    phase_manipulate_eef_pct: float,
    phase_grip_closed_frac: float,
    phase_grip_motion_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-frame bool masks: pause, transport, approach, manipulate (+ dq for debug)."""
    n = state.shape[0]
    dq = _joint_speed(state, joint_dims)
    deef = _eef_speed(state)
    grip_max, dg = _gripper_signals(state)
    active = _pause_mask(dq, pause_percentile, pause_min)
    pause = ~active

    transport = np.zeros(n, dtype=bool)
    approach = np.zeros(n, dtype=bool)
    manipulate = np.zeros(n, dtype=bool)

    if not np.any(active):
        return pause, transport, approach, manipulate, dq

    g_lo = float(grip_max.min())
    g_hi = float(grip_max.max())
    grip_closed = grip_max > (g_lo + phase_grip_closed_frac * (g_hi - g_lo + eps))
    grip_motion_thr = float(np.percentile(dg[active], phase_grip_motion_pct))
    grip_active = dg > grip_motion_thr

    dq_fast = float(np.percentile(dq[active], phase_transport_speed_pct))
    deef_manip_thr = float(np.percentile(deef[active], phase_manipulate_eef_pct))

    transport = active & (dq >= dq_fast) & (~grip_closed)
    manipulate = active & (~transport) & (grip_closed | grip_active) & (deef <= deef_manip_thr)
    approach = active & (~transport) & (~manipulate)
    return pause, transport, approach, manipulate, dq


def _phase_rho(
    state: np.ndarray,
    *,
    joint_dims: int,
    eps: float,
    pause_percentile: float,
    pause_min: float,
    phase_pause_rho: float,
    phase_transport_rho: float,
    phase_approach_rho: float,
    phase_manipulate_rho: float,
    phase_transport_speed_pct: float,
    phase_manipulate_eef_pct: float,
    phase_grip_closed_frac: float,
    phase_grip_motion_pct: float,
) -> np.ndarray:
    """Rule-based phase labels for pick-insert style demos."""
    pause, transport, approach, manipulate, _dq = _phase_masks(
        state,
        joint_dims=joint_dims,
        eps=eps,
        pause_percentile=pause_percentile,
        pause_min=pause_min,
        phase_transport_speed_pct=phase_transport_speed_pct,
        phase_manipulate_eef_pct=phase_manipulate_eef_pct,
        phase_grip_closed_frac=phase_grip_closed_frac,
        phase_grip_motion_pct=phase_grip_motion_pct,
    )
    n = state.shape[0]
    rho = np.full(n, phase_pause_rho, dtype=np.float64)
    rho[transport] = phase_transport_rho
    rho[approach] = phase_approach_rho
    rho[manipulate] = phase_manipulate_rho
    return rho


def episode_phase_names(
    state: np.ndarray,
    *,
    joint_dims: int = 14,
    eps: float = 1e-8,
    pause_percentile: float = 5.0,
    pause_min: float = 1e-4,
    phase_transport_speed_pct: float = 70.0,
    phase_manipulate_eef_pct: float = 65.0,
    phase_grip_closed_frac: float = 0.45,
    phase_grip_motion_pct: float = 60.0,
) -> list[str]:
    pause, transport, approach, manipulate, _dq = _phase_masks(
        state,
        joint_dims=joint_dims,
        eps=eps,
        pause_percentile=pause_percentile,
        pause_min=pause_min,
        phase_transport_speed_pct=phase_transport_speed_pct,
        phase_manipulate_eef_pct=phase_manipulate_eef_pct,
        phase_grip_closed_frac=phase_grip_closed_frac,
        phase_grip_motion_pct=phase_grip_motion_pct,
    )
    names: list[str] = []
    for i in range(state.shape[0]):
        if pause[i]:
            names.append("pause")
        elif transport[i]:
            names.append("transport")
        elif manipulate[i]:
            names.append("manipulate")
        elif approach[i]:
            names.append("approach")
        else:
            names.append("other")
    return names


def compute_episode_rho(
    state: np.ndarray,
    *,
    speed_mode: SpeedMode,
    action: np.ndarray | None,
    joint_dims: int,
    smooth_window: int,
    eps: float,
    pause_percentile: float,
    pause_min: float,
    gripper_weight: float,
    phase_pause_rho: float = 0.5,
    phase_transport_rho: float = 0.0,
    phase_approach_rho: float = 0.55,
    phase_manipulate_rho: float = 1.0,
    phase_transport_speed_pct: float = 70.0,
    phase_manipulate_eef_pct: float = 65.0,
    phase_grip_closed_frac: float = 0.45,
    phase_grip_motion_pct: float = 60.0,
    universal_pause_rho: float = 0.5,
    universal_open_baseline: float = 0.35,
    universal_speed_dq_weight: float = 0.5,
    universal_speed_deef_weight: float = 0.5,
    universal_absolute_speed_ceiling: float = 0.0,
    universal_absolute_transition_fraction: float = 0.4,
    universal_grasp_acquire_penalty: float = 0.70,
    universal_grasp_settle_window: int = 11,
    universal_grasp_trust_floor: float = 0.25,
    universal_grip_quiescent_penalty: float = 0.75,
    universal_grip_activity_window: int = 13,
    normalize_episode: bool = True,
) -> np.ndarray:
    n = state.shape[0]
    if speed_mode == "action_l2":
        if action is None:
            raise ValueError("action_l2 requires action")
        if action.ndim == 3:
            action = action[:, 0, :]
        speed = np.linalg.norm(action, axis=-1).astype(np.float64)
        rho_raw = _sigmoid_slow_score(speed, np.ones(n, dtype=bool), eps)
        return _normalize_episode(_smooth(rho_raw, smooth_window), eps)

    dq = _joint_speed(state, joint_dims)
    active = _pause_mask(dq, pause_percentile, pause_min)

    if speed_mode == "state_joint_delta":
        rho_raw = _sigmoid_slow_score(dq, np.ones(n, dtype=bool), eps)
    elif speed_mode == "v2a":
        rho_raw = _sigmoid_slow_score(dq, active, eps)
    elif speed_mode == "v2b":
        deef = _eef_speed(state)
        grip_score = _gripper_activity(state)
        rho_eef = _sigmoid_slow_score(deef, active, eps)
        rho_grip = np.zeros(n, dtype=np.float64)
        if np.any(active):
            g = grip_score[active]
            g_norm = g / (g.max() + eps)
            rho_grip[active] = g_norm
        rho_raw = np.zeros(n, dtype=np.float64)
        rho_raw[active] = np.maximum(rho_eef[active], gripper_weight * rho_grip[active])
    elif speed_mode == "v3_phase":
        rho_raw = _phase_rho(
            state,
            joint_dims=joint_dims,
            eps=eps,
            pause_percentile=pause_percentile,
            pause_min=pause_min,
            phase_pause_rho=phase_pause_rho,
            phase_transport_rho=phase_transport_rho,
            phase_approach_rho=phase_approach_rho,
            phase_manipulate_rho=phase_manipulate_rho,
            phase_transport_speed_pct=phase_transport_speed_pct,
            phase_manipulate_eef_pct=phase_manipulate_eef_pct,
            phase_grip_closed_frac=phase_grip_closed_frac,
            phase_grip_motion_pct=phase_grip_motion_pct,
        )
        normalize_episode = False
    elif speed_mode in ("v4_wire", "universal"):
        rho_raw = _universal_rho(
            state,
            joint_dims=joint_dims,
            eps=eps,
            pause_percentile=pause_percentile,
            pause_min=pause_min,
            universal_pause_rho=universal_pause_rho,
            universal_open_baseline=universal_open_baseline,
            universal_speed_dq_weight=universal_speed_dq_weight,
            universal_speed_deef_weight=universal_speed_deef_weight,
            universal_absolute_speed_ceiling=universal_absolute_speed_ceiling,
            universal_absolute_transition_fraction=universal_absolute_transition_fraction,
            universal_grasp_acquire_penalty=universal_grasp_acquire_penalty,
            universal_grasp_settle_window=universal_grasp_settle_window,
            universal_grasp_trust_floor=universal_grasp_trust_floor,
            universal_grip_quiescent_penalty=universal_grip_quiescent_penalty,
            universal_grip_activity_window=universal_grip_activity_window,
        )
        normalize_episode = False
    else:
        raise ValueError(f"Unknown speed_mode: {speed_mode}")

    rho_smooth = _smooth(rho_raw, smooth_window)
    if normalize_episode:
        rho_smooth = _normalize_episode(rho_smooth, eps)
    return rho_smooth.astype(np.float32)


@dataclasses.dataclass
class Args:
    repo_id: str
    root: pathlib.Path | None = None
    out_path: pathlib.Path = pathlib.Path("assets/pace/pps/pps_weights.npy")
    speed_mode: SpeedMode = "v2a"
    state_key: str = "observation.state"
    action_key: str = "action"
    joint_dims: int = 14
    smooth_window: int = 7
    eps: float = 1e-8
    pause_percentile: float = 5.0
    pause_min: float = 1e-4
    gripper_weight: float = 0.7
    phase_pause_rho: float = 0.5
    phase_transport_rho: float = 0.0
    phase_approach_rho: float = 0.55
    phase_manipulate_rho: float = 1.0
    phase_transport_speed_pct: float = 70.0
    phase_manipulate_eef_pct: float = 65.0
    phase_grip_closed_frac: float = 0.45
    phase_grip_motion_pct: float = 60.0
    universal_pause_rho: float = 0.5
    universal_open_baseline: float = 0.35
    universal_speed_dq_weight: float = 0.5
    universal_speed_deef_weight: float = 0.5
    universal_absolute_speed_ceiling: float = 0.0
    universal_absolute_transition_fraction: float = 0.4
    universal_grasp_acquire_penalty: float = 0.70
    universal_grasp_settle_window: int = 11
    universal_grasp_trust_floor: float = 0.25
    universal_grip_quiescent_penalty: float = 0.75
    universal_grip_activity_window: int = 13
    # Some embodiments encode 1 as an open gripper.  Universal PPS defines a
    # high grip value as grasp engagement, so flip only the offline proxy input
    # in that case; model observations/actions retain their native convention.
    gripper_open_is_high: bool = False


def main(args: Args) -> None:
    root = args.root or (pathlib.Path.home() / ".cache" / "huggingface" / "lerobot" / args.repo_id)
    with (root / "meta" / "info.json").open() as f:
        info = json.load(f)

    table_paths = _episode_table_paths(root, info)
    if not table_paths:
        raise FileNotFoundError(f"No parquet data files found under {root}")

    columns = ["episode_index", args.state_key]
    if args.speed_mode == "action_l2":
        columns.append(args.action_key)

    total_frames = info["total_frames"]
    rho = np.zeros(total_frames, dtype=np.float32)

    global_idx = 0
    n_episodes = 0

    for path in table_paths:
        table = pq.read_table(path, columns=columns)
        episode_index_col = np.asarray(table["episode_index"].to_pylist())
        state = np.stack(table[args.state_key].to_pylist())
        action = None
        if args.speed_mode == "action_l2":
            action = np.stack(table[args.action_key].to_pylist())

        for ep in np.unique(episode_index_col):
            mask = episode_index_col == ep
            ep_state = state[mask].astype(np.float64)
            if args.gripper_open_is_high:
                if ep_state.shape[1] < 22:
                    raise ValueError(
                        "gripper_open_is_high requires state[..., 20:22] gripper channels, "
                        f"got shape {ep_state.shape}"
                    )
                ep_state = ep_state.copy()
                ep_state[:, 20:22] = 1.0 - ep_state[:, 20:22]
            ep_action = action[mask] if action is not None else None
            n_frames = int(mask.sum())

            rho_ep = compute_episode_rho(
                ep_state,
                speed_mode=args.speed_mode,
                action=ep_action,
                joint_dims=args.joint_dims,
                smooth_window=args.smooth_window,
                eps=args.eps,
                pause_percentile=args.pause_percentile,
                pause_min=args.pause_min,
                gripper_weight=args.gripper_weight,
                phase_pause_rho=args.phase_pause_rho,
                phase_transport_rho=args.phase_transport_rho,
                phase_approach_rho=args.phase_approach_rho,
                phase_manipulate_rho=args.phase_manipulate_rho,
                phase_transport_speed_pct=args.phase_transport_speed_pct,
                phase_manipulate_eef_pct=args.phase_manipulate_eef_pct,
                phase_grip_closed_frac=args.phase_grip_closed_frac,
                phase_grip_motion_pct=args.phase_grip_motion_pct,
                universal_pause_rho=args.universal_pause_rho,
                universal_open_baseline=args.universal_open_baseline,
                universal_speed_dq_weight=args.universal_speed_dq_weight,
                universal_speed_deef_weight=args.universal_speed_deef_weight,
                universal_absolute_speed_ceiling=args.universal_absolute_speed_ceiling,
                universal_absolute_transition_fraction=args.universal_absolute_transition_fraction,
                universal_grasp_acquire_penalty=args.universal_grasp_acquire_penalty,
                universal_grasp_settle_window=args.universal_grasp_settle_window,
                universal_grasp_trust_floor=args.universal_grasp_trust_floor,
                universal_grip_quiescent_penalty=args.universal_grip_quiescent_penalty,
                universal_grip_activity_window=args.universal_grip_activity_window,
            )

            indices = np.where(mask)[0]
            rho[global_idx + indices[0] : global_idx + indices[0] + n_frames] = rho_ep
            n_episodes += 1

        global_idx += len(episode_index_col)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_path, rho)

    print(
        f"[compute_pps_weights] mode={args.speed_mode} joint_dims={args.joint_dims} "
        f"gripper_open_is_high={args.gripper_open_is_high}"
    )
    print(f"  {n_episodes} episodes, {total_frames} frames processed.")
    print(f"  rho stats: mean={rho.mean():.4f}, std={rho.std():.4f}, min={rho.min():.4f}, max={rho.max():.4f}")
    print(f"Wrote: {args.out_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
