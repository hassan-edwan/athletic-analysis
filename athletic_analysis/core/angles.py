"""Joint angles from Halpe-26 keypoint trajectories.

All functions accept (T, 26, 3) arrays and return per-frame angle series in
degrees with NaN where the involved keypoints are low-confidence.

Conventions (2D, view-dependent — assumes roughly sagittal filming for sprint):
- knee / hip / elbow / ankle: interior 3-point angle at the middle joint,
  180 = fully extended.
- trunk_lean: mid-hip -> mid-shoulder segment vs. image vertical, signed so
  positive = leaning in the direction of travel (forward lean).
- thigh_angle_*: thigh segment vs. vertical, positive when the knee is in
  front of the hip relative to travel direction (front-side mechanics).
"""

from __future__ import annotations

import numpy as np

from athletic_analysis.core.pose.skeleton import KP

MIN_CONF = 0.3


def angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at b (degrees) for points shaped (..., 2)."""
    v1 = a - b
    v2 = c - b
    dot = (v1 * v2).sum(axis=-1)
    norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _joint_angle(kpts: np.ndarray, a: str, b: str, c: str) -> np.ndarray:
    pa, pb, pc = kpts[:, KP[a]], kpts[:, KP[b]], kpts[:, KP[c]]
    ang = angle_3pt(pa[:, :2], pb[:, :2], pc[:, :2])
    bad = (pa[:, 2] < MIN_CONF) | (pb[:, 2] < MIN_CONF) | (pc[:, 2] < MIN_CONF)
    ang[bad] = np.nan
    return ang


def travel_direction(kpts: np.ndarray) -> float:
    """Overall horizontal direction of motion: +1 rightward, -1 leftward in image."""
    hip_x = kpts[:, KP["hip_center"], 0]
    conf = kpts[:, KP["hip_center"], 2]
    good = conf >= MIN_CONF
    if good.sum() < 2:
        return 1.0
    x = hip_x[good]
    return 1.0 if x[-1] >= x[0] else -1.0


def _segment_vs_vertical(top: np.ndarray, bottom: np.ndarray, direction: float) -> np.ndarray:
    """Signed angle (deg) of bottom->top segment vs. image vertical 'up'.

    Positive = top point displaced from bottom point in the travel direction.
    Note image y grows downward, so 'up' is -y.
    """
    dx = (top[:, 0] - bottom[:, 0]) * direction
    dy = bottom[:, 1] - top[:, 1]  # positive when top is above bottom
    return np.degrees(np.arctan2(dx, dy))


def compute_angles(kpts: np.ndarray) -> dict[str, np.ndarray]:
    """All angle time-series used by the app. kpts: (T, 26, 3) (smoothed)."""
    kpts = np.asarray(kpts, dtype=np.float64)
    direction = travel_direction(kpts)
    out: dict[str, np.ndarray] = {
        "knee_l": _joint_angle(kpts, "l_hip", "l_knee", "l_ankle"),
        "knee_r": _joint_angle(kpts, "r_hip", "r_knee", "r_ankle"),
        "hip_l": _joint_angle(kpts, "l_shoulder", "l_hip", "l_knee"),
        "hip_r": _joint_angle(kpts, "r_shoulder", "r_hip", "r_knee"),
        "ankle_l": _joint_angle(kpts, "l_knee", "l_ankle", "l_big_toe"),
        "ankle_r": _joint_angle(kpts, "r_knee", "r_ankle", "r_big_toe"),
        "elbow_l": _joint_angle(kpts, "l_shoulder", "l_elbow", "l_wrist"),
        "elbow_r": _joint_angle(kpts, "r_shoulder", "r_elbow", "r_wrist"),
    }

    mid_shoulder = (kpts[:, KP["l_shoulder"], :2] + kpts[:, KP["r_shoulder"], :2]) / 2
    mid_hip = (kpts[:, KP["l_hip"], :2] + kpts[:, KP["r_hip"], :2]) / 2
    trunk = _segment_vs_vertical(mid_shoulder, mid_hip, direction)
    conf = np.minimum(
        np.minimum(kpts[:, KP["l_shoulder"], 2], kpts[:, KP["r_shoulder"], 2]),
        np.minimum(kpts[:, KP["l_hip"], 2], kpts[:, KP["r_hip"], 2]),
    )
    trunk[conf < MIN_CONF] = np.nan
    out["trunk_lean"] = trunk

    for side in ("l", "r"):
        hip = kpts[:, KP[f"{side}_hip"], :2]
        knee = kpts[:, KP[f"{side}_knee"], :2]
        thigh = _segment_vs_vertical(hip, knee, direction)
        # Angle of thigh vs vertical measured from the hip down: recompute with
        # hip as top so positive = knee in front of hip.
        dx = (knee[:, 0] - hip[:, 0]) * direction
        dy = knee[:, 1] - hip[:, 1]  # positive: knee below hip
        thigh = np.degrees(np.arctan2(dx, dy))
        c = np.minimum(kpts[:, KP[f"{side}_hip"], 2], kpts[:, KP[f"{side}_knee"], 2])
        thigh[c < MIN_CONF] = np.nan
        out[f"thigh_{side}"] = thigh

    return out


def estimate_body_height_px(kpts: np.ndarray) -> float:
    """Approximate stature in pixels from median segment lengths (pose-invariant)."""
    def seg(a: str, b: str) -> float:
        pa, pb = kpts[:, KP[a]], kpts[:, KP[b]]
        good = (pa[:, 2] >= MIN_CONF) & (pb[:, 2] >= MIN_CONF)
        if good.sum() < 3:
            return np.nan
        d = np.linalg.norm(pa[good, :2] - pb[good, :2], axis=1)
        return float(np.median(d))

    shank = np.nanmean([seg("l_knee", "l_ankle"), seg("r_knee", "r_ankle")])
    thigh = np.nanmean([seg("l_hip", "l_knee"), seg("r_hip", "r_knee")])
    trunk = np.nanmean([seg("l_hip", "l_shoulder"), seg("r_hip", "r_shoulder")])
    head = seg("neck", "head")
    parts = np.array([shank, thigh, trunk, head])
    if np.isnan(parts[:3]).any():
        return float("nan")
    head_term = parts[3] * 1.3 if np.isfinite(parts[3]) else parts[2] * 0.45
    # Ankle-to-floor offset approximated as 5% of the sum.
    return float((parts[0] + parts[1] + parts[2] + head_term) * 1.05)
