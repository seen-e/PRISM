"""Precision Proxy from Speed (PPS).

This module contains the code-only, framework-independent implementation used
to produce a continuous precision target rho in [0, 1] for each demonstration
frame. It assumes the project state layout documented by :class:`PPSConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PPSConfig:
    """PPS parameters for the dual-arm 22-D state representation.

    State layout: joint positions first, then two XYZ end-effector positions at
    indices [14:20], and two gripper values at [20:22]. Override ``joint_dims``
    only when the leading joint block differs.
    """

    joint_dims: int = 14
    eps: float = 1e-8
    pause_percentile: float = 5.0
    pause_min: float = 1e-4
    pause_rho: float = 0.5
    open_baseline: float = 0.35
    speed_dq_weight: float = 0.5
    speed_deef_weight: float = 0.5
    absolute_speed_ceiling: float = 0.0
    absolute_transition_fraction: float = 0.4
    grasp_acquire_penalty: float = 0.70
    grasp_settle_window: int = 11
    grasp_trust_floor: float = 0.25
    grip_quiescent_penalty: float = 0.75
    grip_activity_window: int = 13
    smooth_window: int = 7


def _validate_state(state: np.ndarray, joint_dims: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2:
        raise ValueError(f"state must have shape [T, D], got {state.shape}")
    if state.shape[0] == 0:
        raise ValueError("state must contain at least one frame")
    if state.shape[1] < 22:
        raise ValueError(f"PPS requires at least 22 state dimensions, got {state.shape[1]}")
    if not 1 <= joint_dims <= 14:
        raise ValueError(f"joint_dims must lie in [1, 14], got {joint_dims}")
    if not np.all(np.isfinite(state)):
        raise ValueError("state contains non-finite values")
    return state


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _causal_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="full")[: len(values)]


def _frame_speed(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(len(values), dtype=np.float64)
    delta = np.linalg.norm(np.diff(values, axis=0), axis=-1)
    return np.concatenate(([delta[0]], delta)).astype(np.float64)


def _joint_speed(state: np.ndarray, joint_dims: int) -> np.ndarray:
    return _frame_speed(state[:, :joint_dims])


def _eef_speed(state: np.ndarray) -> np.ndarray:
    eef = state[:, 14:20].reshape(-1, 2, 3)
    if len(eef) < 2:
        return np.zeros(len(eef), dtype=np.float64)
    per_arm = np.linalg.norm(np.diff(eef, axis=0), axis=2)
    delta = per_arm.max(axis=1)
    return np.concatenate(([delta[0]], delta)).astype(np.float64)


def _gripper_signals(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grip = state[:, 20:22]
    grip_max = grip.max(axis=1)
    return grip_max, _frame_speed(grip)


def _percentile_rank(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values), dtype=np.float64)
    if not np.any(mask):
        return output
    selected = values[mask]
    order = np.argsort(selected, kind="stable")
    ranks = np.empty(len(selected), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(selected))
    output[mask] = ranks
    return output


def compute_episode_pps(state: np.ndarray, config: PPSConfig | None = None) -> np.ndarray:
    """Compute continuous frame-level PPS targets for one episode.

    Slow active motion while the gripper is engaged receives a high score.
    Pauses remain neutral, while grasp acquisition and recent gripper activity
    suppress false precision credit.
    """

    cfg = config or PPSConfig()
    state = _validate_state(state, cfg.joint_dims)
    dq = _joint_speed(state, cfg.joint_dims)
    deef = _eef_speed(state)
    grip_max, dg = _gripper_signals(state)

    pause_threshold = max(float(np.percentile(dq, cfg.pause_percentile)), cfg.pause_min)
    active = dq >= pause_threshold
    rho = np.full(len(state), cfg.pause_rho, dtype=np.float64)
    if not np.any(active):
        return rho.astype(np.float32)

    grip_low = float(grip_max.min())
    grip_high = float(grip_max.max())
    grip_engagement = np.clip((grip_max - grip_low) / (grip_high - grip_low + cfg.eps), 0.0, 1.0)

    weight_sum = cfg.speed_dq_weight + cfg.speed_deef_weight + cfg.eps
    speed = (cfg.speed_dq_weight * dq + cfg.speed_deef_weight * deef) / weight_sum
    slowness = 1.0 - _percentile_rank(speed, active)

    if cfg.absolute_speed_ceiling > 0.0:
        transition = max(cfg.absolute_speed_ceiling * cfg.absolute_transition_fraction, cfg.eps)
        absolute_slow = np.clip((cfg.absolute_speed_ceiling - speed) / transition, 0.0, 1.0)
    else:
        absolute_slow = np.ones_like(speed)

    positive_close = np.maximum(np.diff(grip_engagement, prepend=grip_engagement[0]), 0.0)
    recent_close = _causal_mean(positive_close, cfg.grasp_settle_window)
    close_scale = float(np.percentile(recent_close[active], 92)) + cfg.eps
    grasp_acquisition = np.clip(recent_close / close_scale, 0.0, 1.0)
    trust_acquisition = np.clip(
        1.0 - cfg.grasp_acquire_penalty * grasp_acquisition,
        cfg.grasp_trust_floor,
        1.0,
    )

    recent_grip_activity = _causal_mean(dg, cfg.grip_activity_window)
    activity_scale = float(np.percentile(recent_grip_activity[active], 92)) + cfg.eps
    grip_activity = np.clip(recent_grip_activity / activity_scale, 0.0, 1.0)
    trust_quiescent = np.clip(
        1.0 - cfg.grip_quiescent_penalty * grip_activity,
        cfg.grasp_trust_floor,
        1.0,
    )
    trust = trust_acquisition * trust_quiescent

    active_rho = (
        grip_engagement[active] * slowness[active] * trust[active]
        + (1.0 - grip_engagement[active]) * cfg.open_baseline
    )
    above_neutral = active_rho > cfg.pause_rho
    active_rho[above_neutral] = cfg.pause_rho + (
        active_rho[above_neutral] - cfg.pause_rho
    ) * absolute_slow[active][above_neutral]
    rho[active] = np.clip(active_rho, 0.0, 1.0)
    rho = np.clip(_smooth(rho, cfg.smooth_window), 0.0, 1.0)
    return rho.astype(np.float32)
