"""Jump metrics: height, countermovement depth, takeoff/landing mechanics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from athletic_analysis.core.angles import estimate_body_height_px
from athletic_analysis.core.calibration import Calibration
from athletic_analysis.core.events import JumpPhases
from athletic_analysis.core.pose.skeleton import KP

G = 9.81


@dataclass
class JumpMetrics:
    takeoff_frame: int = -1
    landing_frame: int = -1
    flight_time_s: float = float("nan")
    # Camera-independent: h = g * t^2 / 8 (symmetric flight assumption).
    jump_height_flight_m: float = float("nan")
    # Vertical velocity leaving the ground: v0 = g * t_flight / 2.
    takeoff_velocity_m_s: float = float("nan")
    # Cross-check from hip displacement; meters if calibrated, else BH.
    hip_rise: float = float("nan")
    countermovement_depth: float = float("nan")
    length_unit: str = "BH"
    knee_angle_at_takeoff: float = float("nan")
    hip_angle_at_takeoff: float = float("nan")
    trunk_lean_at_takeoff: float = float("nan")
    peak_knee_flexion_landing: float = float("nan")  # min knee angle after landing
    knee_ankle_sep_ratio_landing: float = float("nan")  # frontal-view valgus proxy


def _at(series: np.ndarray | None, frame: int) -> float:
    if series is None or frame < 0 or frame >= len(series):
        return float("nan")
    return float(series[frame])


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if len(finite) else float("nan")


def compute_jump_metrics(kpts: np.ndarray, angles: dict[str, np.ndarray],
                         jump: JumpPhases | None, fps: float,
                         calib: Calibration | None = None) -> JumpMetrics:
    m = JumpMetrics()
    if jump is None:
        return m
    m.takeoff_frame = jump.takeoff_frame
    m.landing_frame = jump.landing_frame
    m.flight_time_s = (jump.landing_frame - jump.takeoff_frame) / fps
    m.jump_height_flight_m = G * m.flight_time_s ** 2 / 8.0
    m.takeoff_velocity_m_s = G * m.flight_time_s / 2.0

    body_h_px = estimate_body_height_px(kpts)

    def px_to_len(px: float) -> float:
        if calib is not None:
            return calib.to_meters(px)
        if np.isfinite(body_h_px) and body_h_px > 0:
            return px / body_h_px
        return float("nan")

    if calib is not None:
        m.length_unit = "m"

    hip_y = kpts[:, KP["hip_center"], 1].astype(np.float64)
    flight = hip_y[jump.takeoff_frame:jump.landing_frame + 1]
    if len(flight):
        m.hip_rise = px_to_len(jump.baseline_hip_y - float(np.nanmin(flight)))
    m.countermovement_depth = px_to_len(
        float(hip_y[jump.lowest_hip_frame]) - jump.baseline_hip_y)

    knee_l, knee_r = angles.get("knee_l"), angles.get("knee_r")
    m.knee_angle_at_takeoff = _nanmean([
        _at(knee_l, jump.takeoff_frame), _at(knee_r, jump.takeoff_frame)])
    m.hip_angle_at_takeoff = _nanmean([
        _at(angles.get("hip_l"), jump.takeoff_frame),
        _at(angles.get("hip_r"), jump.takeoff_frame)])
    m.trunk_lean_at_takeoff = _at(angles.get("trunk_lean"), jump.takeoff_frame)

    # Landing window: 0.5 s after touchdown.
    end = min(len(hip_y), jump.landing_frame + round(0.5 * fps))
    lo, hi = jump.landing_frame, end
    if hi > lo and knee_l is not None and knee_r is not None:
        stacked = np.vstack([knee_l[lo:hi], knee_r[lo:hi]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns are expected
            window = np.nanmin(stacked, axis=0)
        if np.isfinite(window).any():
            m.peak_knee_flexion_landing = float(np.nanmin(window))
            peak_frame = lo + int(np.nanargmin(window))
            knee_sep = abs(kpts[peak_frame, KP["l_knee"], 0]
                           - kpts[peak_frame, KP["r_knee"], 0])
            ankle_sep = abs(kpts[peak_frame, KP["l_ankle"], 0]
                            - kpts[peak_frame, KP["r_ankle"], 0])
            if ankle_sep > 1e-6:
                m.knee_ankle_sep_ratio_landing = float(knee_sep / ankle_sep)
    return m
