"""Live 'optimal form' overlay: for a posable check (trunk lean, knee angle at
touchdown, front-side thigh) at any frame, compute the athlete's actual value,
the phase-appropriate optimal band, how far off they are, and the geometry of a
*corrected* segment — the same limb rotated to the nearest in-range angle,
anchored at the athlete's real joint. Drawn behind the real skeleton, the gap
between the real limb and this ghost segment IS the deviation.

Non-posable checks (ground contact, cadence, overstride) have no single pose to
correct and are handled by the caller as findings without an overlay.

Angle conventions are taken verbatim from core/angles.py — trunk and thigh are
signed vs. image vertical in the travel direction; knee is the interior
hip-knee-ankle angle. This module inverts those to place the corrected joint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from athletic_analysis.core.coaching import _evaluate, sprint_checks
from athletic_analysis.core.pose.skeleton import KP

POSABLE_KEYS = ("trunk", "knee_strike", "thigh")
_MIN_CONF = 0.3

Pt = tuple[float, float]


@dataclass
class OverlaySpec:
    """Full-frame pixel geometry for drawing one corrected segment."""
    key: str
    anchor: Pt          # joint the segment pivots about (hip / knee)
    actual_end: Pt      # where the athlete's limb actually ends
    optimal_end: Pt     # where it would end at the nearest in-range angle
    head: Pt | None = None       # ghost head above the corrected shoulder (trunk)
    optimal_head: Pt | None = None


@dataclass
class LiveEval:
    """Everything the panel needs at one frame for the active metric."""
    key: str
    value: float
    lo: float
    hi: float
    off: float          # degrees beyond the nearest band edge (0 if in range)
    over: bool          # True = above hi, False = below lo (only meaningful if off>0)
    severity: str       # good | minor | major
    phase: str
    spec: OverlaySpec | None


_ANGLE_KEY = {"trunk": "trunk_lean", "knee_strike": "knee", "thigh": "thigh"}


def phase_at_frame(spans: list[tuple[int, int, str]], frame: int) -> str | None:
    for f0, f1, name in spans:
        if f0 <= frame <= f1:
            return name
    return None


def _conf_ok(kpts_frame: np.ndarray, *names: str) -> bool:
    return all(kpts_frame[KP[n], 2] >= _MIN_CONF for n in names)


def stance_side(kpts_frame: np.ndarray) -> str:
    """Leg in contact = foot nearer the ground = larger ankle y (y grows down)."""
    return "l" if kpts_frame[KP["l_ankle"], 1] >= kpts_frame[KP["r_ankle"], 1] else "r"


def swing_side(kpts_frame: np.ndarray) -> str:
    return "r" if stance_side(kpts_frame) == "l" else "l"


def _target_angle(value: float, lo: float, hi: float) -> float:
    """Nearest in-range angle to `value` — the minimum correction to draw."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _trunk_spec(kpts_frame: np.ndarray, direction: float, target_deg: float
                ) -> OverlaySpec | None:
    if not _conf_ok(kpts_frame, "l_hip", "r_hip", "l_shoulder", "r_shoulder"):
        return None
    mid_hip = (kpts_frame[KP["l_hip"], :2] + kpts_frame[KP["r_hip"], :2]) / 2
    mid_sh = (kpts_frame[KP["l_shoulder"], :2] + kpts_frame[KP["r_shoulder"], :2]) / 2
    length = float(np.linalg.norm(mid_sh - mid_hip))
    theta = math.radians(target_deg)
    opt = (mid_hip[0] + length * math.sin(theta) * direction,
           mid_hip[1] - length * math.cos(theta))
    # Ghost head: extend a bit beyond the corrected shoulder, matching the real
    # neck->head offset length if available.
    head = optimal_head = None
    if _conf_ok(kpts_frame, "head", "neck"):
        head_len = float(np.linalg.norm(
            kpts_frame[KP["head"], :2] - kpts_frame[KP["neck"], :2]))
        head = (float(kpts_frame[KP["head"], 0]), float(kpts_frame[KP["head"], 1]))
        ext = (length + head_len) / max(length, 1e-6)
        optimal_head = (mid_hip[0] + length * ext * math.sin(theta) * direction,
                        mid_hip[1] - length * ext * math.cos(theta))
    return OverlaySpec("trunk", (float(mid_hip[0]), float(mid_hip[1])),
                       (float(mid_sh[0]), float(mid_sh[1])), opt,
                       head=head, optimal_head=optimal_head)


def _thigh_spec(kpts_frame: np.ndarray, side: str, direction: float,
                target_deg: float) -> OverlaySpec | None:
    if not _conf_ok(kpts_frame, f"{side}_hip", f"{side}_knee"):
        return None
    hip = kpts_frame[KP[f"{side}_hip"], :2]
    knee = kpts_frame[KP[f"{side}_knee"], :2]
    length = float(np.linalg.norm(knee - hip))
    theta = math.radians(target_deg)
    opt = (hip[0] + length * math.sin(theta) * direction,
           hip[1] + length * math.cos(theta))
    return OverlaySpec("thigh", (float(hip[0]), float(hip[1])),
                       (float(knee[0]), float(knee[1])), opt)


def _knee_spec(kpts_frame: np.ndarray, side: str, target_deg: float
               ) -> OverlaySpec | None:
    if not _conf_ok(kpts_frame, f"{side}_hip", f"{side}_knee", f"{side}_ankle"):
        return None
    hip = kpts_frame[KP[f"{side}_hip"], :2]
    knee = kpts_frame[KP[f"{side}_knee"], :2]
    ankle = kpts_frame[KP[f"{side}_ankle"], :2]
    shank_len = float(np.linalg.norm(ankle - knee))
    thigh_ang = math.atan2(hip[1] - knee[1], hip[0] - knee[0])  # knee->hip
    actual_ang = math.atan2(ankle[1] - knee[1], ankle[0] - knee[0])
    phi = math.radians(target_deg)
    # Interior angle between knee->hip and knee->ankle should be `phi`; two
    # candidate shank directions, pick the one closest to the real shank.
    cands = [thigh_ang + phi, thigh_ang - phi]
    best = min(cands, key=lambda a: abs(math.atan2(math.sin(a - actual_ang),
                                                   math.cos(a - actual_ang))))
    opt = (knee[0] + shank_len * math.cos(best), knee[1] + shank_len * math.sin(best))
    return OverlaySpec("knee_strike", (float(knee[0]), float(knee[1])),
                       (float(ankle[0]), float(ankle[1])), opt)


def live_eval(key: str, frame: int, kpts: np.ndarray, angles: dict,
              spans: list[tuple[int, int, str]], level: str,
              direction: float) -> LiveEval | None:
    """Evaluate the active posable metric at `frame`; None for unknown/unusable
    keys or frames outside any graded phase."""
    if key not in POSABLE_KEYS:
        return None
    phase = phase_at_frame(spans, frame)
    if phase is None:
        return None
    checks = dict(sprint_checks(level).get(phase, []))
    check = checks.get(key)
    if check is None:  # e.g. cadence/thigh only defined in some phases
        return None
    kf = kpts[frame]

    if key == "trunk":
        value = float(angles["trunk_lean"][frame])
        side = None
    elif key == "knee_strike":
        side = stance_side(kf)
        value = float(angles[f"knee_{side}"][frame])
    else:  # thigh
        side = swing_side(kf)
        value = float(angles[f"thigh_{side}"][frame])
    if not np.isfinite(value):
        return None

    finding = _evaluate(check, value, phase, frame)
    severity = finding.severity if finding else "good"
    lo, hi = check.lo, check.hi
    off = max(lo - value, value - hi, 0.0)
    over = value > hi
    target = _target_angle(value, lo, hi)

    if key == "trunk":
        spec = _trunk_spec(kf, direction, target)
    elif key == "knee_strike":
        spec = _knee_spec(kf, side, target)
    else:
        spec = _thigh_spec(kf, side, direction, target)

    return LiveEval(key=key, value=value, lo=lo, hi=hi, off=off, over=over,
                    severity=severity, phase=phase, spec=spec)


# --- drawing ------------------------------------------------------------------

# BGR. The optimal ("where you should be") ghost is a bright mint green —
# semantically "good", and light/saturated enough to stand apart from the
# medium-green left-leg skeleton color. The connector spanning the gap from
# your real limb to the optimal one is amber (minor) or red (major), so the
# size AND color of the gap both read as the deviation.
_OPTIMAL = (170, 255, 170)
_GAP = {"good": (120, 235, 120), "minor": (40, 180, 235), "major": (60, 70, 235)}


def draw_overlay(image: np.ndarray, spec: OverlaySpec, severity: str,
                 off_deg: float) -> np.ndarray:
    """Draw the corrected 'optimal' segment (ghost) + a gap connector from the
    athlete's real limb end to the optimal one, on a full-frame BGR image.
    In-range frames draw nothing (the athlete is already on target)."""
    if severity == "good":
        return image
    scale = max(image.shape[0], image.shape[1]) / 1000.0
    thick = max(2, round(3 * scale))

    def px(p: Pt) -> tuple[int, int]:
        return (int(round(p[0])), int(round(p[1])))

    # Blend the optimal segment so it reads as a ghost behind the real skeleton.
    ghost = image.copy()
    cv2.line(ghost, px(spec.anchor), px(spec.optimal_end), _OPTIMAL, thick + 1,
             cv2.LINE_AA)
    cv2.circle(ghost, px(spec.optimal_end), max(3, round(4 * scale)), _OPTIMAL,
               -1, cv2.LINE_AA)
    cv2.addWeighted(ghost, 0.6, image, 0.4, 0, image)

    # The gap: real limb end -> optimal end, in the severity color, with the
    # degrees-off label at its midpoint.
    gap_color = _GAP.get(severity, _GAP["minor"])
    cv2.line(image, px(spec.actual_end), px(spec.optimal_end), gap_color,
             max(1, round(2 * scale)), cv2.LINE_AA)
    if off_deg >= 1.0:
        mid = ((spec.actual_end[0] + spec.optimal_end[0]) / 2 + 8 * scale,
               (spec.actual_end[1] + spec.optimal_end[1]) / 2)
        label = f"{off_deg:.0f} deg off"
        org = (int(mid[0]), int(mid[1]))
        cv2.putText(image, label, org, cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.4, 0.5 * scale), (0, 0, 0), max(2, round(3 * scale)),
                    cv2.LINE_AA)
        cv2.putText(image, label, org, cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.4, 0.5 * scale), gap_color, max(1, round(scale)),
                    cv2.LINE_AA)
    return image


def major_frames(key: str, kpts: np.ndarray, angles: dict,
                 spans: list[tuple[int, int, str]], level: str,
                 direction: float) -> list[int]:
    """Frames where the active metric is a *major* deviation — the scrub-bar
    markers for 'significantly off target'."""
    out = []
    for frame in range(len(kpts)):
        ev = live_eval(key, frame, kpts, angles, spans, level, direction)
        if ev is not None and ev.severity == "major":
            out.append(frame)
    return out
