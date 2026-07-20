"""Schematic 'textbook' stick figure at a target joint angle — the fallback
reference for compare.py when no real step in the clip pulls off a check
(every step shares the same fault, so there's no real footage of "good" to
show). Deliberately simple: a hand-built forward-kinematics figure, not the
26-point Halpe skeleton — the goal is a clear, immediately-readable diagram
of one joint at one angle, not biomechanical realism.

Only checks with a single-frame joint angle are posable this way (see
compare.POSABLE_KEYS): trunk lean, knee angle at touchdown, and front-side
thigh angle. Timing/distance checks (ground contact time, cadence,
overstride) can't be drawn as a static pose and never reach this module.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

CANVAS_W, CANVAS_H = 220, 320
GROUND_Y = 285
HIP = (108, 165)
TRUNK_LEN = 82
THIGH_LEN = 74
SHANK_LEN = 68
HEAD_R = 13

_INK = (235, 235, 235)      # BGR near-white: trunk/head/stance leg
_FOCUS = (60, 210, 255)     # BGR amber: the leg/segment the check is about
_BG = (18, 19, 23)


def _at_angle(origin: tuple[float, float], length: float, angle_deg: float,
             direction: int = 1) -> tuple[float, float]:
    """Point `length` away from `origin`, `angle_deg` off vertical (0 = straight
    down), tilting toward +x when direction=1 — the same convention
    core/angles.py uses for trunk lean and thigh angle."""
    rad = math.radians(angle_deg)
    return (origin[0] + length * math.sin(rad) * direction,
            origin[1] + length * math.cos(rad))


def _bend(knee: tuple[float, float], thigh_dir: tuple[float, float],
         interior_deg: float, shank_len: float, mirror: bool = False) -> tuple[float, float]:
    """Ankle position giving an interior hip-knee-ankle angle of `interior_deg`.
    `thigh_dir` is the knee->hip vector. An interior angle alone has two valid
    solutions (mirrored across the thigh axis) — `mirror` picks which one
    draws a sensible leg for the given context (forward-reaching touchdown
    vs. a folded swing-leg shank)."""
    hx, hy = thigh_dir
    base = math.degrees(math.atan2(hy, hx))
    sign = -1.0 if mirror else 1.0
    theta = math.radians(base + sign * interior_deg)
    return (knee[0] + shank_len * math.cos(theta),
            knee[1] + shank_len * math.sin(theta))


def _draw_leg(img: np.ndarray, hip: tuple, knee: tuple, ankle: tuple,
             color: tuple, thickness: int = 4) -> None:
    for a, b in ((hip, knee), (knee, ankle)):
        cv2.line(img, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])),
                 color, thickness, cv2.LINE_AA)
    cv2.circle(img, (round(knee[0]), round(knee[1])), 4, color, -1, cv2.LINE_AA)
    # A short foot so the leg reads as a leg, not a bare line.
    foot = (ankle[0] + 18, ankle[1] + 2)
    cv2.line(img, (round(ankle[0]), round(ankle[1])), (round(foot[0]), round(foot[1])),
             color, thickness, cv2.LINE_AA)


def _blank_canvas() -> np.ndarray:
    img = np.full((CANVAS_H, CANVAS_W, 3), _BG, dtype=np.uint8)
    cv2.line(img, (10, GROUND_Y), (CANVAS_W - 10, GROUND_Y), (60, 61, 68), 2,
             cv2.LINE_AA)
    return img


def _draw_trunk_and_head(img: np.ndarray, hip: tuple, lean_deg: float,
                         color: tuple) -> tuple:
    neck = _at_angle(hip, TRUNK_LEN, lean_deg)
    head = _at_angle(hip, TRUNK_LEN + HEAD_R + 8, lean_deg)
    cv2.line(img, (round(hip[0]), round(hip[1])), (round(neck[0]), round(neck[1])),
             color, 5, cv2.LINE_AA)
    cv2.circle(img, (round(head[0]), round(head[1])), HEAD_R, color, -1, cv2.LINE_AA)
    return neck


def render(key: str, target_deg: float, unit_label: str) -> np.ndarray:
    """A labeled schematic BGR image of the target angle for `key`
    ("trunk" | "knee_strike" | "thigh"). Raises ValueError for other keys —
    callers should check compare.POSABLE_KEYS first."""
    img = _blank_canvas()
    stance_ankle = (HIP[0] - 4, GROUND_Y)
    stance_knee = ((HIP[0] + stance_ankle[0]) / 2 - 6, HIP[1] + (GROUND_Y - HIP[1]) * 0.55)

    if key == "trunk":
        _draw_trunk_and_head(img, HIP, target_deg, _FOCUS)
        _draw_leg(img, HIP, stance_knee, stance_ankle, _INK)
        front_ankle = (HIP[0] + 30, GROUND_Y - 10)
        front_knee = (HIP[0] + 18, HIP[1] + 40)
        _draw_leg(img, HIP, front_knee, front_ankle, _INK)
    elif key == "knee_strike":
        _draw_trunk_and_head(img, HIP, 8.0, _INK)
        _draw_leg(img, HIP, stance_knee, stance_ankle, _INK)
        knee = _at_angle(HIP, THIGH_LEN, 4.0)
        thigh_dir = (HIP[0] - knee[0], HIP[1] - knee[1])
        ankle = _bend(knee, thigh_dir, target_deg, SHANK_LEN)
        _draw_leg(img, HIP, knee, ankle, _FOCUS)
    elif key == "thigh":
        _draw_trunk_and_head(img, HIP, 10.0, _INK)
        _draw_leg(img, HIP, stance_knee, stance_ankle, _INK)
        knee = _at_angle(HIP, THIGH_LEN, target_deg)
        thigh_dir = (HIP[0] - knee[0], HIP[1] - knee[1])
        ankle = _bend(knee, thigh_dir, 55.0, SHANK_LEN * 0.7, mirror=True)  # folded, mid-swing
        _draw_leg(img, HIP, knee, ankle, _FOCUS)
    else:
        raise ValueError(f"reference_pose.render: no schematic for key {key!r}")

    label = f"target {target_deg:.0f}{unit_label}"
    cv2.putText(img, label, (10, CANVAS_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
               (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (10, CANVAS_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
               (235, 235, 235), 1, cv2.LINE_AA)
    return img
