"""Detection of gait and jump events from foot/hip trajectories.

Image coordinates: y grows downward, so the ground is at *high* y and being
airborne means *low* foot y. Thresholds are scaled by each signal's own
amplitude so detection works at any video resolution and framing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from athletic_analysis.core.pose.skeleton import KP


@dataclass
class GaitEvent:
    frame: int
    side: str  # "left" | "right"
    kind: str  # "strike" | "toeoff"


@dataclass
class JumpPhases:
    takeoff_frame: int
    landing_frame: int
    lowest_hip_frame: int  # countermovement bottom (max image-y of hip before takeoff)
    baseline_hip_y: float  # standing hip height in px (image y)


def foot_y_signal(kpts: np.ndarray, side: str) -> np.ndarray:
    """Vertical position of the foot: mean of big toe and heel."""
    prefix = "l_" if side == "left" else "r_"
    toe = kpts[:, KP[prefix + "big_toe"], 1]
    heel = kpts[:, KP[prefix + "heel"], 1]
    return (toe + heel) / 2.0


def contact_mask(foot_y: np.ndarray, fps: float,
                 ground_band: float = 0.15, speed_factor: float = 2.5,
                 min_contact_s: float = 0.04, merge_gap_s: float = 0.03) -> np.ndarray:
    """Boolean mask of frames where this foot is on the ground.

    Contact = foot near the ground level (top `ground_band` fraction of its
    vertical travel) AND moving slowly (|vy| below `speed_factor` amplitudes/s;
    swing-phase foot speed is several times higher).
    """
    y = np.asarray(foot_y, dtype=np.float64)
    n = len(y)
    if n < 3:
        return np.zeros(n, dtype=bool)
    lo, hi = np.nanpercentile(y, 5), np.nanpercentile(y, 95)
    amp = hi - lo
    if amp <= 1e-9:  # foot never moves: treat as always on the ground
        return np.ones(n, dtype=bool)
    ground = np.nanpercentile(y, 90)
    near_ground = y > ground - ground_band * amp
    vy = np.gradient(y) * fps
    slow = np.abs(vy) < speed_factor * amp
    mask = near_ground & slow

    # Merge sub-gap holes, then drop implausibly short contacts.
    mask = _close_gaps(mask, max(1, round(merge_gap_s * fps)))
    mask = _drop_short_runs(mask, max(2, round(min_contact_s * fps)))
    return mask


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def _close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    inv_runs = _runs(~mask)
    for s, e in inv_runs:
        if s == 0 or e == len(mask):
            continue  # keep leading/trailing gaps
        if e - s <= max_gap:
            out[s:e] = True
    return out


def _drop_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = mask.copy()
    for s, e in _runs(mask):
        if e - s < min_len:
            out[s:e] = False
    return out


def detect_gait_events(kpts: np.ndarray, fps: float) -> list[GaitEvent]:
    """Foot strikes and toe-offs for both feet, sorted by frame."""
    events: list[GaitEvent] = []
    for side in ("left", "right"):
        mask = contact_mask(foot_y_signal(kpts, side), fps)
        for s, e in _runs(mask):
            if s > 0:
                events.append(GaitEvent(frame=int(s), side=side, kind="strike"))
            if e < len(mask):
                events.append(GaitEvent(frame=int(e - 1), side=side, kind="toeoff"))
    events.sort(key=lambda ev: ev.frame)
    return events


def detect_jump(kpts: np.ndarray, fps: float,
                min_flight_s: float = 0.15) -> JumpPhases | None:
    """Find the main jump: the airborne interval (both feet off ground) with the
    greatest hip rise. Returns None if no plausible flight phase exists."""
    left = contact_mask(foot_y_signal(kpts, "left"), fps)
    right = contact_mask(foot_y_signal(kpts, "right"), fps)
    airborne = ~left & ~right
    hip_y = kpts[:, KP["hip_center"], 1].astype(np.float64)

    min_len = max(2, round(min_flight_s * fps))
    baseline_n = max(2, round(0.5 * fps))
    baseline_hip = float(np.nanmedian(hip_y[:baseline_n]))

    best: tuple[float, tuple[int, int]] | None = None
    for s, e in _runs(airborne):
        if e - s < min_len or s == 0 or e >= len(airborne):
            continue
        rise = baseline_hip - float(np.nanmin(hip_y[s:e]))  # px above standing
        if best is None or rise > best[0]:
            best = (rise, (s, e))
    if best is None:
        return None
    s, e = best[1]
    takeoff, landing = int(s - 1), int(e)
    pre = hip_y[:takeoff + 1]
    lowest = int(np.nanargmax(pre)) if len(pre) else takeoff
    return JumpPhases(takeoff_frame=takeoff, landing_frame=landing,
                      lowest_hip_frame=lowest, baseline_hip_y=baseline_hip)
