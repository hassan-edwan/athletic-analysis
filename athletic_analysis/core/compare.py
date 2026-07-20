"""Pairs each form fault with a real example of the athlete doing it well.

For a phase/metric flagged by coaching.py (the median across steps in that
phase was out of range), find the individual step in that same phase whose
value actually sits inside the optimal band — real footage of "what good
looks like" for this exact athlete, not a stock photo. When no step in the
clip pulls it off (every step shares the fault), there's nothing real to
show; `reference_pose.py` covers that fallback with a schematic diagram.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from athletic_analysis.core.coaching import (FormFinding, PhaseBucket,
                                             _Check, sprint_checks)

# Checks with a single-frame joint angle a stick figure can depict. Timing/
# distance checks (contact time, cadence, overstride) can't be shown as one
# static pose, so they never get a synthetic-figure fallback — only a real
# best-step comparison, or nothing.
POSABLE_KEYS = {"trunk", "knee_strike", "thigh"}


@dataclass
class StepComparison:
    finding: FormFinding
    check: _Check
    best_frame: int | None  # a real step of this athlete's, in range; None if none exists
    best_value: float | None
    posable: bool  # whether reference_pose.py can draw a synthetic fallback


def _deviation(value: float, check: _Check) -> float:
    if not np.isfinite(value):
        return float("inf")
    if check.lo <= value <= check.hi:
        return 0.0
    return min(abs(value - check.lo), abs(value - check.hi))


def build_comparisons(findings: list[FormFinding],
                      buckets: dict[str, PhaseBucket],
                      level: str = "trained") -> list[StepComparison]:
    """One StepComparison per non-good finding that has a real coaching check
    behind it (skips findings whose key/phase we can't re-look-up, which
    shouldn't happen in practice but keeps this defensive)."""
    checks_by_phase = sprint_checks(level)
    out: list[StepComparison] = []
    for f in findings:
        if f.severity == "good" or not f.key:
            continue
        check = next((c for k, c in checks_by_phase.get(f.phase, []) if k == f.key),
                     None)
        if check is None:
            continue
        bucket = buckets.get(f.phase)
        values = bucket.values.get(f.key, []) if bucket else []
        frames = bucket.strike_frames if bucket else []
        best_frame, best_value, best_dev = None, None, float("inf")
        for value, frame in zip(values, frames):
            if frame == f.frame or not np.isfinite(value):
                continue
            dev = _deviation(value, check)
            if dev < best_dev:
                best_dev, best_frame, best_value = dev, frame, value
        has_real = best_frame is not None and best_dev == 0.0
        out.append(StepComparison(
            finding=f, check=check,
            best_frame=best_frame if has_real else None,
            best_value=best_value if has_real else None,
            posable=f.key in POSABLE_KEYS))
    return out
