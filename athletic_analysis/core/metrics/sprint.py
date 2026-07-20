"""Sprint metrics derived from gait events + angle series.

Distances are reported in meters when a Calibration is set, otherwise in
body-heights (BH) using the pose-based stature estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from athletic_analysis.core.angles import estimate_body_height_px
from athletic_analysis.core.calibration import Calibration
from athletic_analysis.core.events import GaitEvent
from athletic_analysis.core.pose.skeleton import KP
from athletic_analysis.core.velocity import rolling_nanmean


@dataclass
class StepRecord:
    side: str
    strike_frame: int
    toeoff_frame: int | None
    contact_time_s: float
    flight_time_s: float  # toe-off to next (opposite) strike; NaN if unknown
    step_time_s: float  # this strike to next strike; NaN for last step
    step_length: float  # m or BH; NaN if unknown
    step_speed: float  # step_length / step_time: m/s or BH/s; NaN if unknown
    knee_angle_at_strike: float
    swing_thigh_angle: float  # front-side mechanics: opposite thigh vs vertical at strike
    trunk_lean_at_strike: float


@dataclass
class SprintMetrics:
    steps: list[StepRecord] = field(default_factory=list)
    cadence_spm: float = float("nan")  # steps per minute
    mean_contact_s: float = float("nan")
    mean_flight_s: float = float("nan")
    mean_step_length: float = float("nan")
    mean_speed: float = float("nan")  # m/s or BH/s over the detected steps
    max_speed: float = float("nan")  # peak 0.4 s-averaged horizontal speed
    mean_trunk_lean_deg: float = float("nan")
    length_unit: str = "BH"  # "m" when calibrated
    body_height_px: float = float("nan")


def _at(series: np.ndarray, frame: int) -> float:
    if series is None or frame < 0 or frame >= len(series):
        return float("nan")
    return float(series[frame])


def compute_sprint_metrics(kpts: np.ndarray, angles: dict[str, np.ndarray],
                           events: list[GaitEvent], fps: float,
                           calib: Calibration | None = None) -> SprintMetrics:
    m = SprintMetrics()
    strikes = [ev for ev in events if ev.kind == "strike"]
    if not strikes:
        return m

    body_h_px = estimate_body_height_px(kpts)
    m.body_height_px = body_h_px
    if calib is not None:
        m.length_unit = "m"

    def px_to_len(px: float) -> float:
        if calib is not None:
            return calib.to_meters(px)
        if np.isfinite(body_h_px) and body_h_px > 0:
            return px / body_h_px
        return float("nan")

    hip_x = kpts[:, KP["hip_center"], 0].astype(np.float64)

    for i, strike in enumerate(strikes):
        side_key = strike.side[0]  # 'l' / 'r'
        other = "r" if side_key == "l" else "l"
        # Matching toe-off: first toe-off of the same foot after this strike.
        toeoff = next((ev for ev in events
                       if ev.kind == "toeoff" and ev.side == strike.side
                       and ev.frame > strike.frame), None)
        nxt = strikes[i + 1] if i + 1 < len(strikes) else None

        contact_s = ((toeoff.frame - strike.frame) / fps
                     if toeoff and (nxt is None or toeoff.frame <= nxt.frame)
                     else float("nan"))
        flight_s = ((nxt.frame - toeoff.frame) / fps
                    if toeoff and nxt and nxt.frame > toeoff.frame
                    else float("nan"))
        step_s = (nxt.frame - strike.frame) / fps if nxt else float("nan")
        step_len = (px_to_len(abs(hip_x[nxt.frame] - hip_x[strike.frame]))
                    if nxt else float("nan"))
        step_speed = step_len / step_s if nxt and step_s > 0 else float("nan")

        m.steps.append(StepRecord(
            side=strike.side,
            strike_frame=strike.frame,
            toeoff_frame=toeoff.frame if toeoff else None,
            contact_time_s=contact_s,
            flight_time_s=flight_s,
            step_time_s=step_s,
            step_length=step_len,
            step_speed=step_speed,
            knee_angle_at_strike=_at(angles.get(f"knee_{side_key}"), strike.frame),
            swing_thigh_angle=_at(angles.get(f"thigh_{other}"), strike.frame),
            trunk_lean_at_strike=_at(angles.get("trunk_lean"), strike.frame),
        ))

    def nanmean(values: list[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")

    mean_step_time = nanmean([s.step_time_s for s in m.steps])
    if np.isfinite(mean_step_time) and mean_step_time > 0:
        m.cadence_spm = 60.0 / mean_step_time
    first, last = strikes[0].frame, strikes[-1].frame
    if last > first:
        m.mean_speed = px_to_len(abs(hip_x[last] - hip_x[first])) / ((last - first) / fps)
        vx = np.abs(np.gradient(hip_x)) * fps
        smooth_vx = rolling_nanmean(vx, max(3, round(0.4 * fps)))
        window = smooth_vx[first:last + 1]
        if np.isfinite(window).any():
            m.max_speed = px_to_len(float(np.nanmax(window)))
    m.mean_contact_s = nanmean([s.contact_time_s for s in m.steps])
    m.mean_flight_s = nanmean([s.flight_time_s for s in m.steps])
    m.mean_step_length = nanmean([s.step_length for s in m.steps])
    m.mean_trunk_lean_deg = nanmean([s.trunk_lean_at_strike for s in m.steps])
    return m
