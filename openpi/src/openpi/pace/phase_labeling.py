"""Gripper-closure based phase segmentation.

Shared by the D2 offline diagnostic (proposal.md Sec. 18.5) and the CAS
phase-sigma target computation (proposal.md Sec. 19, Phase 2). An episode of
the bimanual pick -> handoff -> insert task is split into three phases using
only the gripper channels of `observation.state` -- no manual labels:

  1. smooth each gripper channel with a short moving average (denoise);
  2. detect the first "sustained closure" event on either gripper
     -> pick/handoff boundary;
  3. detect the second sustained closure event -> handoff/insert boundary;
  4. fall back to fixed 1/3, 2/3 split points if fewer than 2 closure events
     are found (e.g. a demo where handoff does not involve a full open/close
     cycle).

This heuristic was validated in the D2 diagnostic: sigma_j(insert) computed
from these labels is consistently smaller than sigma_j(pick/handoff) on the
fine (low-scale) action dimensions, and dropping this segmentation in favor
of a global sigma masks the precision-collapse effect entirely (see Sec.
18.5 "已知局限" for remaining caveats).
"""

from __future__ import annotations

import numpy as np

PHASES = ("pick", "handoff", "insert")


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float64)
    kernel = np.ones(window) / window
    pad = window // 2
    xp = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def _sustained_closure_events(gripper: np.ndarray, close_thresh: float, sustain_frames: int) -> list[int]:
    """Frame indices where `gripper` transitions into a sustained closed state."""
    closed = gripper > close_thresh
    events: list[int] = []
    run = 0
    for t, c in enumerate(closed):
        run = run + 1 if c else 0
        if run == sustain_frames:
            events.append(t - sustain_frames + 1)
    return events


def detect_phase_boundaries(
    gripper_left: np.ndarray,
    gripper_right: np.ndarray,
    *,
    smooth_window: int = 7,
    sustain_frames: int = 5,
    close_thresh: float | None = None,
) -> tuple[int, int]:
    """Returns (pick_end, handoff_end) frame indices (exclusive upper bound).

    Phase assignment: pick = [0, pick_end), handoff = [pick_end, handoff_end),
    insert = [handoff_end, T).
    """
    n = len(gripper_left)
    if n < 4:
        # Degenerate/very short episode: split as evenly as possible.
        return max(1, n // 3), max(2, 2 * n // 3)

    gl = _smooth(np.asarray(gripper_left), smooth_window)
    gr = _smooth(np.asarray(gripper_right), smooth_window)

    if close_thresh is None:
        lo = min(gl.min(), gr.min())
        hi = max(gl.max(), gr.max())
        close_thresh = lo + 0.5 * (hi - lo) if hi > lo else lo

    events = sorted(
        _sustained_closure_events(gl, close_thresh, sustain_frames)
        + _sustained_closure_events(gr, close_thresh, sustain_frames)
    )

    if len(events) >= 2:
        pick_end, handoff_end = events[0], events[1]
    elif len(events) == 1:
        pick_end = events[0]
        handoff_end = pick_end + (n - pick_end) // 2
    else:
        pick_end = n // 3
        handoff_end = 2 * n // 3

    pick_end = int(np.clip(pick_end, 1, n - 2))
    handoff_end = int(np.clip(handoff_end, pick_end + 1, n - 1))
    return pick_end, handoff_end


def label_phases(gripper_left: np.ndarray, gripper_right: np.ndarray, **kwargs) -> np.ndarray:
    """Returns an object array of shape (T,) with labels in {pick, handoff, insert}."""
    n = len(gripper_left)
    pick_end, handoff_end = detect_phase_boundaries(gripper_left, gripper_right, **kwargs)
    labels = np.empty(n, dtype=object)
    labels[:pick_end] = "pick"
    labels[pick_end:handoff_end] = "handoff"
    labels[handoff_end:] = "insert"
    return labels
