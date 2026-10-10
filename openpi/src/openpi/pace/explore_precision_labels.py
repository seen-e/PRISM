"""Experimental precision labels (rho) for PPS exploration.

Does NOT replace production ``compute_pps_weights.py``. Each method maps an
episode state trajectory to rho in [0, 1] (per-episode min-max unless noted).

Methods
-------
speed_universal
    Current production universal speed proxy (baseline).
phase_v3
    Rule-based transport / approach / manipulate / pause (v3_phase).
phase_gripper
    Gripper-closure segmentation: pick / handoff / insert.
physical_tolerance
    Task-space tolerance shrinks near the episode goal EEF pose.
jacobian_tolerance
    Jacobian-projected tolerance: precision rises when local J amplifies
    joint errors into large task-space motion (data-estimated J).
local_success_sens
    Proxy for |d(success metric)/d(motion)| using goal distance + gripper.
jacobian_x_speed
    Geometric mean of jacobian_tolerance and speed_universal (hybrid).
shuffled
    speed_universal with per-episode temporal structure destroyed (control).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from openpi.pace.compute_pps_weights import (
    _eef_speed,
    _gripper_signals,
    _joint_speed,
    _normalize_episode,
    _pause_mask,
    _percentile_rank,
    _phase_rho,
    _smooth,
    _universal_rho,
)
from openpi.pace.phase_labeling import label_phases

MethodName = Literal[
    "speed_universal",
    "phase_v3",
    "phase_gripper",
    "physical_tolerance",
    "jacobian_tolerance",
    "local_success_sens",
    "jacobian_x_speed",
    "shuffled",
]

ALL_METHODS: tuple[MethodName, ...] = (
    "speed_universal",
    "phase_v3",
    "phase_gripper",
    "physical_tolerance",
    "jacobian_tolerance",
    "local_success_sens",
    "jacobian_x_speed",
    "shuffled",
)

METHOD_DESCRIPTIONS: dict[str, str] = {
    "speed_universal": "Baseline: slow active motion while grasping (production universal PPS).",
    "phase_v3": "Discrete phase rules: transport / approach / manipulate / pause.",
    "phase_gripper": "Gripper-closure phases: pick / handoff / insert.",
    "physical_tolerance": "High rho when task-space error budget is small (near goal EEF).",
    "jacobian_tolerance": "High rho when local J amplifies joint motion into task space.",
    "local_success_sens": "High rho when success proxy changes quickly per unit motion.",
    "jacobian_x_speed": "sqrt(speed_universal * jacobian_tolerance) hybrid.",
    "shuffled": "Control: speed_universal rho shuffled within episode.",
}


def _episode_rank01(values: np.ndarray, active: np.ndarray | None = None, eps: float = 1e-8) -> np.ndarray:
    mask = np.ones(values.shape[0], dtype=bool) if active is None else active
    ranked = _percentile_rank(values, mask)
    if not np.any(mask):
        return np.zeros_like(values)
    v = ranked[mask]
    return np.clip((ranked - v.min()) / (v.max() - v.min() + eps), 0.0, 1.0)


def rho_speed_universal(state: np.ndarray, *, joint_dims: int = 14, smooth_window: int = 7, **kw) -> np.ndarray:
    universal_kw = {
        "pause_percentile": kw.get("pause_percentile", 5.0),
        "pause_min": kw.get("pause_min", 1e-4),
        **{k: v for k, v in kw.items() if k.startswith("universal_")},
    }
    rho = _universal_rho(state, joint_dims=joint_dims, eps=kw.get("eps", 1e-8), **universal_kw)
    if smooth_window > 1:
        rho = _smooth(rho, smooth_window)
    return np.clip(rho, 0.0, 1.0).astype(np.float32)


def rho_phase_v3(state: np.ndarray, *, joint_dims: int = 14, smooth_window: int = 1, **kw) -> np.ndarray:
    eps = kw.get("eps", 1e-8)
    rho = _phase_rho(
        state,
        joint_dims=joint_dims,
        eps=eps,
        pause_percentile=kw.get("pause_percentile", 5.0),
        pause_min=kw.get("pause_min", 1e-4),
        phase_pause_rho=kw.get("phase_pause_rho", 0.5),
        phase_transport_rho=kw.get("phase_transport_rho", 0.0),
        phase_approach_rho=kw.get("phase_approach_rho", 0.55),
        phase_manipulate_rho=kw.get("phase_manipulate_rho", 1.0),
        phase_transport_speed_pct=kw.get("phase_transport_speed_pct", 70.0),
        phase_manipulate_eef_pct=kw.get("phase_manipulate_eef_pct", 65.0),
        phase_grip_closed_frac=kw.get("phase_grip_closed_frac", 0.45),
        phase_grip_motion_pct=kw.get("phase_grip_motion_pct", 60.0),
    )
    if smooth_window > 1:
        rho = _smooth(rho, smooth_window)
    return np.clip(rho, 0.0, 1.0).astype(np.float32)


def rho_phase_gripper(state: np.ndarray, *, smooth_window: int = 1, **kw) -> np.ndarray:
    labels = label_phases(state[:, 20], state[:, 21])
    rho_map = {
        "pick": float(kw.get("gripper_pick_rho", 0.12)),
        "handoff": float(kw.get("gripper_handoff_rho", 0.45)),
        "insert": float(kw.get("gripper_insert_rho", 0.95)),
    }
    rho = np.array([rho_map[str(l)] for l in labels], dtype=np.float64)
    if smooth_window > 1:
        rho = _smooth(rho, smooth_window)
    return np.clip(rho, 0.0, 1.0).astype(np.float32)


def rho_physical_tolerance(state: np.ndarray, *, joint_dims: int = 14, smooth_window: int = 7, **kw) -> np.ndarray:
    """Precision cost when allowed task-space error is small (near episode goal)."""
    eps = kw.get("eps", 1e-8)
    eef = state[:, 14:20]
    goal = eef[-1]
    dist = np.linalg.norm(eef - goal, axis=1)
    # Allowed error grows with distance to goal; invert for precision cost.
    tol = dist / (float(dist.max()) + eps)
    precision = 1.0 - tol

    grip_max, _ = _gripper_signals(state)
    g_lo, g_hi = float(grip_max.min()), float(grip_max.max())
    grip_eng = np.clip((grip_max - g_lo) / (g_hi - g_lo + eps), 0.0, 1.0)

    dq = _joint_speed(state, joint_dims)
    active = _pause_mask(dq, kw.get("pause_percentile", 5.0), kw.get("pause_min", 1e-4))
    rho = np.full(state.shape[0], 0.5, dtype=np.float64)
    rho[active] = np.clip(precision[active] * (0.35 + 0.65 * grip_eng[active]), 0.0, 1.0)
    if smooth_window > 1:
        rho = _smooth(rho, smooth_window)
    return np.clip(rho, 0.0, 1.0).astype(np.float32)


def _local_jacobian_gain(state: np.ndarray, *, joint_dims: int, window: int, eps: float) -> np.ndarray:
    """Estimate scalar task amplification ||Δx|| / ||Δq|| in a sliding window."""
    joints = state[:, :joint_dims]
    eef = state[:, 14:20]
    n = state.shape[0]
    gains = np.zeros(n, dtype=np.float64)
    half = max(1, window // 2)
    for t in range(n):
        a, b = max(0, t - half), min(n, t + half + 1)
        dQ = np.diff(joints[a:b], axis=0)
        dX = np.diff(eef[a:b], axis=0)
        if len(dQ) < 2:
            continue
        dq_norm = np.linalg.norm(dQ, axis=1)
        dx_norm = np.linalg.norm(dX, axis=1)
        valid = dq_norm > eps
        if not np.any(valid):
            continue
        gains[t] = float(np.median(dx_norm[valid] / (dq_norm[valid] + eps)))
    return gains


def rho_jacobian_tolerance(state: np.ndarray, *, joint_dims: int = 14, smooth_window: int = 7, **kw) -> np.ndarray:
    """Precision rises when joint motion is amplified in task space (high local J gain)."""
    eps = kw.get("eps", 1e-8)
    window = int(kw.get("jacobian_window", 9))
    gains = _local_jacobian_gain(state, joint_dims=joint_dims, window=window, eps=eps)

    eef = state[:, 14:20]
    goal = eef[-1]
    dist = np.linalg.norm(eef - goal, axis=1)
    inv_tol = 1.0 / (dist / (float(dist.max()) + eps) + 0.05)

    dq = _joint_speed(state, joint_dims)
    active = _pause_mask(dq, kw.get("pause_percentile", 5.0), kw.get("pause_min", 1e-4))
    gain_rank = _episode_rank01(gains, active, eps=eps)
    tol_rank = _episode_rank01(inv_tol, active, eps=eps)
    rho = np.full(state.shape[0], 0.5, dtype=np.float64)
    rho[active] = np.clip(0.55 * gain_rank[active] + 0.45 * tol_rank[active], 0.0, 1.0)
    if smooth_window > 1:
        rho = _smooth(rho, smooth_window)
    return np.clip(rho, 0.0, 1.0).astype(np.float32)


def rho_local_success_sens(state: np.ndarray, *, joint_dims: int = 14, smooth_window: int = 7, **kw) -> np.ndarray:
    """Proxy: how fast a success metric changes relative to arm motion."""
    eps = kw.get("eps", 1e-8)
    eef = state[:, 14:20]
    goal = eef[-1]
    grip_max, _ = _gripper_signals(state)
    g_lo, g_hi = float(grip_max.min()), float(grip_max.max())
    grip_eng = np.clip((grip_max - g_lo) / (g_hi - g_lo + eps), 0.0, 1.0)

    dist = np.linalg.norm(eef - goal, axis=1)
    success = -dist + 0.35 * grip_eng
    ds = np.abs(np.gradient(success))

    dq = _joint_speed(state, joint_dims)
    deef = _eef_speed(state)
    motion = 0.4 * dq + 0.6 * deef + eps
    sensitivity = ds / motion

    active = _pause_mask(dq, kw.get("pause_percentile", 5.0), kw.get("pause_min", 1e-4))
    rho = np.full(state.shape[0], 0.5, dtype=np.float64)
    rho[active] = _episode_rank01(sensitivity, active, eps=eps)[active]
    if smooth_window > 1:
        rho = _smooth(rho, smooth_window)
    return np.clip(rho, 0.0, 1.0).astype(np.float32)


def rho_jacobian_x_speed(state: np.ndarray, **kw) -> np.ndarray:
    a = rho_speed_universal(state, **kw).astype(np.float64)
    b = rho_jacobian_tolerance(state, **kw).astype(np.float64)
    return np.clip(np.sqrt(a * b), 0.0, 1.0).astype(np.float32)


def rho_shuffled(state: np.ndarray, *, seed: int = 0, **kw) -> np.ndarray:
    base = rho_speed_universal(state, **kw)
    rng = np.random.default_rng(seed)
    out = base.copy()
    rng.shuffle(out)
    return out.astype(np.float32)


_COMPUTE_FNS = {
    "speed_universal": rho_speed_universal,
    "phase_v3": rho_phase_v3,
    "phase_gripper": rho_phase_gripper,
    "physical_tolerance": rho_physical_tolerance,
    "jacobian_tolerance": rho_jacobian_tolerance,
    "local_success_sens": rho_local_success_sens,
    "jacobian_x_speed": rho_jacobian_x_speed,
    "shuffled": rho_shuffled,
}


def compute_episode_rho(state: np.ndarray, method: MethodName, **kwargs) -> np.ndarray:
    if method not in _COMPUTE_FNS:
        raise ValueError(f"Unknown method {method!r}; choose from {ALL_METHODS}")
    return _COMPUTE_FNS[method](state, **kwargs)


def method_summary(state: np.ndarray, method: MethodName, **kwargs) -> dict[str, float]:
    rho = compute_episode_rho(state, method, **kwargs)
    return {
        "mean": float(rho.mean()),
        "std": float(rho.std()),
        "min": float(rho.min()),
        "max": float(rho.max()),
    }


def pairwise_correlation(state: np.ndarray, methods: tuple[MethodName, ...] | None = None, **kwargs) -> np.ndarray:
    names = methods or ALL_METHODS
    rhos = np.stack([compute_episode_rho(state, m, **kwargs) for m in names], axis=0)
    return np.corrcoef(rhos)
