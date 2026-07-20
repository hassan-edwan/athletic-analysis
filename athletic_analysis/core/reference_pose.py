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

Sized and weighted to be legible at the ~150-200px thumbnail height it's
actually displayed at (compare_panel.py), not just in an isolated test
render — verify by re-running `tools/render_reference_poses.py` (or the
equivalent ad hoc script) and looking at the PNG before trusting this file;
the interior-angle math in `_bend()` has bitten this exact module once
already (a wrong `mirror` choice sent an ankle flying up past the head).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

CANVAS_W, CANVAS_H = 340, 480
GROUND_Y = 442
HIP = (167, 256)
TRUNK_LEN = 127
THIGH_LEN = 115
SHANK_LEN = 105
HEAD_R = 20

_INK = (225, 225, 225)      # BGR near-white: trunk/head/stance leg
_FOCUS = (60, 210, 255)     # BGR amber: the leg/segment the check is about
_ARC = (255, 255, 255)      # BGR white: the angle arc + label, reads over either color
_BG = (18, 19, 23)
_GROUND = (60, 61, 68)

_LEG_THICKNESS = 7
_TRUNK_THICKNESS = 9
_JOINT_R = 6
_ARC_RADIUS = 56


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


def _cv_angle(dx: float, dy: float) -> float:
    """Direction vector -> the angle convention cv2.ellipse expects (degrees,
    0 = +x axis, increasing clockwise in image coordinates)."""
    return math.degrees(math.atan2(dy, dx))


def _put_label(img: np.ndarray, text: str, anchor: tuple[float, float],
               scale: float = 0.55) -> None:
    """Text with a dark outline so it reads over either the ink or focus
    color, or the background."""
    pt = (round(anchor[0]), round(anchor[1]))
    cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX, scale, _ARC, 1, cv2.LINE_AA)


def _draw_angle_arc(img: np.ndarray, vertex: tuple[float, float],
                    theta1_cv: float, theta2_cv: float, degree_text: str,
                    radius: float = _ARC_RADIUS) -> None:
    """Arc + degree label at `vertex`, sweeping the *minor* angle between two
    cv2.ellipse-convention directions (always the readable short way around,
    never the reflex angle)."""
    lo, hi = sorted(((theta1_cv + 360) % 360, (theta2_cv + 360) % 360))
    if hi - lo > 180:
        lo, hi = hi - 360, lo
    center = (round(vertex[0]), round(vertex[1]))
    cv2.ellipse(img, center, (round(radius), round(radius)), 0, lo, hi,
               _ARC, 2, cv2.LINE_AA)
    bisector = math.radians((lo + hi) / 2)
    label_r = radius + 26
    anchor = (vertex[0] + label_r * math.cos(bisector) - 16,
             vertex[1] + label_r * math.sin(bisector) + 6)
    _put_label(img, degree_text, anchor)


def _draw_leg(img: np.ndarray, hip: tuple, knee: tuple, ankle: tuple,
             color: tuple) -> None:
    for a, b in ((hip, knee), (knee, ankle)):
        cv2.line(img, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])),
                 color, _LEG_THICKNESS, cv2.LINE_AA)
    cv2.circle(img, (round(knee[0]), round(knee[1])), _JOINT_R, color, -1, cv2.LINE_AA)
    # A short foot so the leg reads as a leg, not a bare line.
    foot = (ankle[0] + 26, ankle[1] + 3)
    cv2.line(img, (round(ankle[0]), round(ankle[1])), (round(foot[0]), round(foot[1])),
             color, _LEG_THICKNESS, cv2.LINE_AA)


def _blank_canvas() -> np.ndarray:
    img = np.full((CANVAS_H, CANVAS_W, 3), _BG, dtype=np.uint8)
    cv2.line(img, (14, GROUND_Y), (CANVAS_W - 14, GROUND_Y), _GROUND, 3, cv2.LINE_AA)
    return img


def _draw_trunk_and_head(img: np.ndarray, hip: tuple, lean_deg: float,
                         color: tuple) -> tuple:
    neck = _at_angle(hip, TRUNK_LEN, lean_deg)
    head = _at_angle(hip, TRUNK_LEN + HEAD_R + 10, lean_deg)
    cv2.line(img, (round(hip[0]), round(hip[1])), (round(neck[0]), round(neck[1])),
             color, _TRUNK_THICKNESS, cv2.LINE_AA)
    cv2.circle(img, (round(head[0]), round(head[1])), HEAD_R, color, -1, cv2.LINE_AA)
    return neck


def render(key: str, target_deg: float, unit_label: str) -> np.ndarray:
    """A labeled schematic BGR image of the target angle for `key`
    ("trunk" | "knee_strike" | "thigh"). Raises ValueError for other keys —
    callers should check compare.POSABLE_KEYS first."""
    img = _blank_canvas()
    stance_ankle = (HIP[0] - 6, GROUND_Y)
    stance_knee = ((HIP[0] + stance_ankle[0]) / 2 - 10,
                   HIP[1] + (GROUND_Y - HIP[1]) * 0.55)
    degree_text = f"{target_deg:.0f}{unit_label}"

    if key == "trunk":
        _draw_leg(img, HIP, stance_knee, stance_ankle, _INK)
        front_ankle = (HIP[0] + 46, GROUND_Y - 16)
        front_knee = (HIP[0] + 28, HIP[1] + 62)
        _draw_leg(img, HIP, front_knee, front_ankle, _INK)
        _draw_trunk_and_head(img, HIP, target_deg, _FOCUS)
        # Angle at the hip, between "straight up" (the vertical reference,
        # 0 deg) and the actual trunk direction.
        _draw_angle_arc(img, HIP, 90, 90 - target_deg, degree_text)
    elif key == "knee_strike":
        _draw_trunk_and_head(img, HIP, 8.0, _INK)
        _draw_leg(img, HIP, stance_knee, stance_ankle, _INK)
        knee = _at_angle(HIP, THIGH_LEN, 4.0)
        thigh_dir = (HIP[0] - knee[0], HIP[1] - knee[1])
        ankle = _bend(knee, thigh_dir, target_deg, SHANK_LEN)
        _draw_leg(img, HIP, knee, ankle, _FOCUS)
        # Angle at the knee itself, between the thigh (knee->hip) and shank
        # (knee->ankle) — the actual joint the check measures.
        shank_dir = (ankle[0] - knee[0], ankle[1] - knee[1])
        _draw_angle_arc(img, knee, _cv_angle(*thigh_dir), _cv_angle(*shank_dir),
                        degree_text, radius=40)
    elif key == "thigh":
        _draw_trunk_and_head(img, HIP, 10.0, _INK)
        _draw_leg(img, HIP, stance_knee, stance_ankle, _INK)
        knee = _at_angle(HIP, THIGH_LEN, target_deg)
        thigh_dir = (HIP[0] - knee[0], HIP[1] - knee[1])
        ankle = _bend(knee, thigh_dir, 55.0, SHANK_LEN * 0.7, mirror=True)  # folded, mid-swing
        _draw_leg(img, HIP, knee, ankle, _FOCUS)
        # Angle at the hip, between "straight down" (0 deg) and the thigh —
        # front-side knee lift is measured the same way trunk lean is.
        _draw_angle_arc(img, HIP, 90, 90 - target_deg, degree_text)
    else:
        raise ValueError(f"reference_pose.render: no schematic for key {key!r}")

    return img


# --- animated fallback ----------------------------------------------------
# A short, natural-reading motion arc around the target angle — approach
# from a bit off-target, settle exactly on it, hold, ease slightly past and
# recover — rather than one frozen pose. Offsets are added to target_deg;
# the settle/hold segments (where offset is 0) are what a viewer mostly
# sees since they're the longest-held frames in the loop.
_MOTION_KEYFRAMES: list[tuple[float, float]] = [
    (0.00, -15.0),  # approaching
    (0.35, 0.0),    # arrives at target
    (0.55, 0.0),    # holds
    (0.75, 6.0),    # eases slightly past
    (1.00, 0.0),    # recovers to target
]


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _motion_offset(t: float) -> float:
    """Interpolated angle offset at normalized time t in [0, 1]."""
    for (t0, a0), (t1, a1) in zip(_MOTION_KEYFRAMES, _MOTION_KEYFRAMES[1:]):
        if t0 <= t <= t1:
            local = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return a0 + (a1 - a0) * _smoothstep(local)
    return _MOTION_KEYFRAMES[-1][1]


def render_sequence(key: str, target_deg: float, unit_label: str,
                    n_frames: int = 12) -> list[np.ndarray]:
    """A short animated loop around `target_deg` — the synthetic-figure
    equivalent of a real slow-motion replay clip, for compare_panel.py's
    ReplayClip widget when no real step in the clip qualifies as a match.
    Each frame is a full `render()` call, so the label always matches what's
    actually drawn, including through the approach/overshoot frames."""
    frames = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        frames.append(render(key, target_deg + _motion_offset(t), unit_label))
    return frames
