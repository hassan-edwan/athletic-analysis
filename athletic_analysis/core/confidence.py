"""Honest, heuristic confidence scoring for measured metrics.

A metric's trustworthiness depends on more than the pose model's keypoint
scores. The dominant hidden limiter for timing metrics is temporal resolution:
an elite ~90 ms ground contact at 30 fps spans only ~3 frames (±33%). This
module folds the signals we can defensibly estimate into a coarse
High / Medium / Low rating and names the single limiting factor, so the UI can
say e.g. "Medium — limited by 30 fps" rather than presenting every number with
equal authority.

Deliberately coarse: three buckets, not false decimals. Never invents
certainty — uncalibrated distances are capped at Medium (relative, not wrong),
and we never claim to verify the 2D filming plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from athletic_analysis.core.angles import MIN_CONF
from athletic_analysis.core.pose.skeleton import KP

HIGH, MEDIUM, LOW = "High", "Medium", "Low"

# Frames an event needs before timing is trustworthy; below this, temporal
# adequacy scales down linearly.
_TARGET_EVENT_FRAMES = 7
_FULL_STEP_SAMPLE = 4


def _level(score: float) -> str:
    if score >= 0.75:
        return HIGH
    if score >= 0.5:
        return MEDIUM
    return LOW


@dataclass
class MetricConfidence:
    score: float
    level: str
    limiter: str  # human-readable dominant limiter ("" when High/unlimited)


def _combine(factors: dict[str, float]) -> MetricConfidence:
    """Overall = product of factors; the smallest factor is the named limiter."""
    if not factors:
        return MetricConfidence(1.0, HIGH, "")
    score = float(np.prod(list(factors.values())))
    worst_key = min(factors, key=factors.get)
    limiter = "" if factors[worst_key] >= 0.75 else worst_key
    return MetricConfidence(score, _level(score), limiter)


def detection_factor(kpts: np.ndarray, joints: list[str],
                     frames: list[int]) -> float:
    """Mean keypoint confidence of `joints` over `frames` (0..1)."""
    if kpts is None or not joints or not frames:
        return 1.0
    idx = [KP[j] for j in joints if j in KP]
    frames = [f for f in frames if 0 <= f < len(kpts)]
    if not idx or not frames:
        return 1.0
    conf = kpts[np.ix_(frames, idx)][:, :, 2]
    return float(np.clip(np.nanmean(conf), 0.0, 1.0))


def temporal_factor(frames_spanned: float) -> float:
    """How well the frame rate resolves an event of this many frames."""
    if not np.isfinite(frames_spanned) or frames_spanned <= 0:
        return 0.4
    return float(np.clip(frames_spanned / _TARGET_EVENT_FRAMES, 0.25, 1.0))


def sample_factor(n: int) -> float:
    """Trust grows with the number of contributing reps/steps."""
    if n <= 0:
        return 0.4
    return float(np.clip(n / _FULL_STEP_SAMPLE, 0.4, 1.0))


# Named factor keys double as the limiter text shown to the user.
_F_DETECT = "joint tracking"
_F_TIME = "frame rate"
_F_SAMPLE = "few steps"
_F_CALIB = "not calibrated"
_F_PLAUSIBLE = "tracking noise"


def metric_confidence(*, detection: float | None = None,
                      frames_spanned: float | None = None,
                      n_samples: int | None = None,
                      plausibility: float | None = None,
                      uncalibrated_distance: bool = False) -> MetricConfidence:
    """Assemble a metric's confidence from whichever signals apply.

    `plausibility` (0..1) is the rigid-bone consistency of the frames this
    metric was read from — a low value means the pose was physically
    implausible there (jitter or an L/R label swap), so the number is shaky
    regardless of the model's own keypoint scores."""
    factors: dict[str, float] = {}
    if detection is not None:
        factors[_F_DETECT] = detection
    if frames_spanned is not None:
        factors[_F_TIME] = temporal_factor(frames_spanned)
    if n_samples is not None:
        factors[_F_SAMPLE] = sample_factor(n_samples)
    if plausibility is not None:
        factors[_F_PLAUSIBLE] = float(np.clip(plausibility, 0.0, 1.0))
    result = _combine(factors)
    # Uncalibrated distance is relative, not wrong: cap at Medium, don't multiply.
    if uncalibrated_distance and result.level == HIGH:
        return MetricConfidence(min(result.score, 0.74), MEDIUM, _F_CALIB)
    return result


@dataclass
class ClipQuality:
    detection_rate: float  # fraction of frames with a credible person
    fps: float
    fps_adequate: bool
    calibrated: bool
    level: str
    notes: list[str]


def clip_quality(kpts: np.ndarray | None, fps: float, calibrated: bool,
                 mean_plausibility: float | None = None,
                 view: str | None = None) -> ClipQuality:
    """Overall analysis-quality badge for the whole clip.

    `mean_plausibility` folds rigid-bone tracking noise into the score;
    `view` (e.g. "frontal"/"oblique") is surfaced as an informational note —
    per-metric validity by camera plane is handled where each metric is
    graded, since a plane that's wrong for sprint angles can be right for a
    frontal knee-valgus check."""
    if kpts is None or len(kpts) == 0:
        return ClipQuality(0.0, fps, False, calibrated, LOW,
                           ["No pose data."])
    # Person considered detected in a frame if the hip is above threshold.
    hip_conf = kpts[:, KP["hip_center"], 2]
    detection_rate = float(np.mean(hip_conf >= MIN_CONF))
    fps_adequate = fps >= 60.0
    notes: list[str] = []
    notes.append(f"{detection_rate * 100:.0f}% frames tracked")
    if not fps_adequate:
        notes.append(f"{fps:.0f} fps limits contact/flight timing "
                     f"(60+ recommended)")
    if not calibrated:
        notes.append("uncalibrated — distances/speeds in body-heights")

    plaus_factor = 1.0
    if mean_plausibility is not None and np.isfinite(mean_plausibility):
        plaus_factor = float(np.clip(mean_plausibility, 0.0, 1.0))
        if plaus_factor < 0.9:
            notes.append(f"{(1 - plaus_factor) * 100:.0f}% of frames look "
                         "mistracked (jitter or L/R swap)")
    if view is not None and view != "sagittal":
        notes.append(f"filmed ~{view} — some angle metrics assume a side view")

    det_factor = float(np.clip(detection_rate, 0.0, 1.0))
    fps_factor = 1.0 if fps_adequate else 0.6
    score = det_factor * fps_factor * plaus_factor
    return ClipQuality(detection_rate, fps, fps_adequate, calibrated,
                       _level(score), notes)
