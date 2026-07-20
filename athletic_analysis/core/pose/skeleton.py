"""Halpe-26 keypoint layout, bone connections, and OpenCV overlay rendering.

One renderer serves both the live UI (frames are composed with OpenCV before
display) and annotated-video export, so they always look identical.
"""

from __future__ import annotations

import cv2
import numpy as np

HALPE26_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
    "head", "neck", "hip_center",
    "l_big_toe", "r_big_toe", "l_small_toe", "r_small_toe", "l_heel", "r_heel",
]

KP = {name: i for i, name in enumerate(HALPE26_NAMES)}

BONES = [
    ("head", "neck"), ("nose", "head"),
    ("neck", "l_shoulder"), ("neck", "r_shoulder"), ("neck", "hip_center"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("hip_center", "l_hip"), ("hip_center", "r_hip"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
    ("l_ankle", "l_heel"), ("l_ankle", "l_big_toe"), ("l_big_toe", "l_small_toe"),
    ("r_ankle", "r_heel"), ("r_ankle", "r_big_toe"), ("r_big_toe", "r_small_toe"),
]

# BGR colors: left limb green, right limb orange, axial white-blue.
_LEFT = (80, 200, 80)
_RIGHT = (60, 140, 255)
_CENTER = (230, 200, 160)


def _bone_color(a: str, b: str) -> tuple[int, int, int]:
    if a.startswith("l_") or b.startswith("l_"):
        return _LEFT
    if a.startswith("r_") or b.startswith("r_"):
        return _RIGHT
    return _CENTER


def draw_pose(frame: np.ndarray, kpts: np.ndarray, conf_thresh: float = 0.3) -> np.ndarray:
    """Draw skeleton in place on a BGR frame. kpts: (26, 3)."""
    if kpts is None:
        return frame
    scale = max(frame.shape[0], frame.shape[1]) / 1000.0
    thickness = max(1, round(2 * scale))
    radius = max(2, round(3 * scale))
    for a, b in BONES:
        pa, pb = kpts[KP[a]], kpts[KP[b]]
        if pa[2] < conf_thresh or pb[2] < conf_thresh:
            continue
        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                 _bone_color(a, b), thickness, cv2.LINE_AA)
    for name, i in KP.items():
        x, y, c = kpts[i]
        if c < conf_thresh:
            continue
        cv2.circle(frame, (int(x), int(y)), radius, _bone_color(name, name), -1, cv2.LINE_AA)
    return frame


def draw_angle_labels(frame: np.ndarray, kpts: np.ndarray, angles: dict[str, float],
                      conf_thresh: float = 0.3) -> np.ndarray:
    """Write joint-angle values next to their joints. angles: name -> degrees (NaN ok)."""
    anchor = {
        "knee_l": "l_knee", "knee_r": "r_knee",
        "hip_l": "l_hip", "hip_r": "r_hip",
        "ankle_l": "l_ankle", "ankle_r": "r_ankle",
        "elbow_l": "l_elbow", "elbow_r": "r_elbow",
    }
    scale = max(frame.shape[0], frame.shape[1]) / 1400.0
    font_scale = max(0.4, 0.5 * scale)
    for name, value in angles.items():
        joint = anchor.get(name)
        if joint is None or not np.isfinite(value):
            continue
        x, y, c = kpts[KP[joint]]
        if c < conf_thresh:
            continue
        text = f"{value:.0f}"
        org = (int(x) + 8, int(y) - 6)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if "trunk_lean" in angles and np.isfinite(angles["trunk_lean"]):
        draw_info_text(frame, f"trunk {angles['trunk_lean']:+.0f} deg", row=0)
    return frame


def draw_info_text(frame: np.ndarray, text: str, row: int = 0) -> np.ndarray:
    """Status line in the top-left corner; `row` stacks multiple lines."""
    org = (12, 28 + row * 26)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame
