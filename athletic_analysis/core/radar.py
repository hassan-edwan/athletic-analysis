"""Sprint factor profile: score five sprint-mechanics factors 0–100 for the
pentagon (radar) chart.

Scores are normalized against the same level-tiered optimal bands the form
grader uses (sprint_checks), via the shared phase bucketing, so the radar can
never disagree with the findings table: a metric graded "good" scores 100,
the minor zone maps to 60–100, major to below 60, bottoming out at zero three
tolerances beyond the band. Axes with no usable data in the clip (e.g. no
max-velocity steps in a start-only clip) stay NaN rather than pretending.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from athletic_analysis.core.coaching import (_fmt_range, _fmt_value,
                                             bucket_sprint_steps, median_of,
                                             sprint_checks)
from athletic_analysis.core.metrics.sprint import SprintMetrics, StepRecord

RADAR_AXES = ("Stiffness / contact", "Front-side mechanics", "Posture / trunk",
              "Foot placement", "Rhythm")

# Axis -> metric keys that feed it (cadence handled with symmetry in Rhythm).
_AXIS_KEYS = {
    "Stiffness / contact": ("contact_ms",),
    "Front-side mechanics": ("thigh",),
    "Posture / trunk": ("trunk",),
    "Foot placement": ("overstride", "knee_strike"),
    "Rhythm": ("cadence",),
}

# Symmetry index at or beyond which the symmetry sub-score hits zero.
_SYM_ZERO = 0.15


@dataclass
class AxisScore:
    name: str
    score: float  # 0..100; NaN = no data
    detail: str
    n_steps: int = 0


@dataclass
class SprintRadar:
    axes: list[AxisScore] = field(default_factory=list)  # RADAR_AXES order
    overall: float = float("nan")
    level: str = "trained"


def band_score(value: float, lo: float, hi: float, tol: float) -> float:
    """Map a measured value to 0–100 against an optimal band, anchored to the
    grader's severity semantics: in band = 100, minor zone 100→60, major zone
    60→0 (floor at three tolerances beyond the band)."""
    if not np.isfinite(value):
        return float("nan")
    d = max(lo - value, value - hi, 0.0)
    if d == 0.0:
        return 100.0
    if tol <= 0:
        return 0.0
    if d <= tol:
        return 100.0 - 40.0 * (d / tol)
    return max(0.0, 60.0 - 60.0 * ((d - tol) / (2.0 * tol)))


def symmetry_score(steps: list[StepRecord]) -> float:
    """0–100 left/right symmetry from per-side medians of step time and
    contact time; NaN when neither quantity has a finite value per side."""
    subs = []
    for attr in ("step_time_s", "contact_time_s"):
        left = median_of([getattr(s, attr) for s in steps if s.side == "left"])
        right = median_of([getattr(s, attr) for s in steps if s.side == "right"])
        mean = 0.5 * (left + right)
        if not (np.isfinite(left) and np.isfinite(right)) or mean <= 0:
            continue
        si = abs(left - right) / mean
        subs.append(100.0 * float(np.clip(1.0 - si / _SYM_ZERO, 0.0, 1.0)))
    return float(np.mean(subs)) if subs else float("nan")


def compute_sprint_radar(kpts: np.ndarray, sprint: SprintMetrics | None,
                         velocities: dict[str, np.ndarray], fps: float,
                         level: str = "trained") -> SprintRadar | None:
    if sprint is None or not sprint.steps:
        return None
    checks_by_phase = sprint_checks(level)
    buckets = bucket_sprint_steps(kpts, sprint, velocities, fps)

    radar = SprintRadar(level=level)
    for name in RADAR_AXES:
        # (score, weight, detail) per contributing (phase, metric).
        contributions: list[tuple[float, float, str]] = []
        for phase, bucket in buckets.items():
            checks = dict(checks_by_phase.get(phase, []))
            for key in _AXIS_KEYS[name]:
                check = checks.get(key)
                if check is None:
                    continue
                value = median_of(bucket.values.get(key, []))
                score = band_score(value, check.lo, check.hi, check.tol)
                if not np.isfinite(score):
                    continue
                detail = (f"{check.metric.lower()} "
                          f"{_fmt_value(value, check.unit)} vs "
                          f"{_fmt_range(check)} ({phase})")
                contributions.append((score, len(bucket.strike_frames), detail))
        if name == "Rhythm":
            sym = symmetry_score(sprint.steps)
            if np.isfinite(sym):
                contributions.append(
                    (sym, len(sprint.steps),
                     f"L/R symmetry {sym:.0f}/100 over {len(sprint.steps)} steps"))
        if contributions:
            weights = np.array([w for _s, w, _d in contributions])
            scores = np.array([s for s, _w, _d in contributions])
            axis_score = float(np.average(scores, weights=weights))
            dominant = max(contributions, key=lambda c: c[1])[2]
            detail = "; ".join(dict.fromkeys(
                [dominant] + [d for _s, _w, d in contributions]))
            n_steps = int(max(w for _s, w, _d in contributions))
        else:
            axis_score, detail, n_steps = float("nan"), "no data in this clip", 0
        radar.axes.append(AxisScore(name=name, score=axis_score,
                                    detail=detail, n_steps=n_steps))

    finite = [a.score for a in radar.axes if np.isfinite(a.score)]
    radar.overall = float(np.mean(finite)) if finite else float("nan")
    return radar
