"""Self-awareness signals: is the pose track trustworthy, and is the camera
even in a plane where the metrics mean anything?

Two independent checks the rest of the pipeline can gate on:

- `tracking_quality`: rigid bones (thigh, shank, upper arm, forearm, trunk
  side) have essentially constant length over a clip. Frames where a bone's
  length jumps are pose errors — jitter, or the classic 2D near/far leg
  label-swap — and shouldn't be trusted equally. Produces a per-frame
  plausibility score and the suspect frames.

- `classify_view`: sagittal / frontal / oblique from how far apart the left
  and right shoulders/hips sit horizontally relative to stature. Sagittal
  filming projects them nearly on top of each other; frontal spreads them a
  shoulder-width apart. Lets the app refuse to report, say, a frontal-plane
  knee-valgus proxy from a side view.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from athletic_analysis.core.angles import MIN_CONF, estimate_body_height_px
from athletic_analysis.core.pose.skeleton import KP

# Rigid segments whose length should hold constant (toes/head excluded: short,
# noisy, or not rigid relative to the body).
_RIGID_BONES = [
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_hip", "l_shoulder"), ("r_hip", "r_shoulder"),
]

SAGITTAL, FRONTAL, OBLIQUE = "sagittal", "frontal", "oblique"


@dataclass
class TrackingQuality:
    plausibility: np.ndarray            # (T,) fraction of rigid bones OK, 0..1
    mean_plausibility: float
    suspect_frames: list[int] = field(default_factory=list)


def tracking_quality(kpts: np.ndarray, tol: float = 0.35,
                     gross_tol: float = 0.5,
                     suspect_below: float = 0.6) -> TrackingQuality:
    """Per-frame plausibility from rigid-bone length constancy. A bone counts
    as OK on a frame when both endpoints are confident and its length is within
    `tol` of that bone's clip-median; plausibility is the fraction of
    (measurable) rigid bones OK. A frame is *suspect* when its plausibility is
    low OR any single bone is grossly wrong (beyond `gross_tol`) — a lone
    keypoint flying off (a jitter spike or L/R label swap) is an error even if
    the other nine bones are fine."""
    kpts = np.asarray(kpts, dtype=np.float64)
    T = len(kpts)
    if T == 0:
        return TrackingQuality(np.zeros(0), 1.0, [])

    ok_counts = np.zeros(T)
    measurable = np.zeros(T)
    gross = np.zeros(T, dtype=bool)
    for a, b in _RIGID_BONES:
        pa, pb = kpts[:, KP[a]], kpts[:, KP[b]]
        good = (pa[:, 2] >= MIN_CONF) & (pb[:, 2] >= MIN_CONF)
        length = np.linalg.norm(pa[:, :2] - pb[:, :2], axis=1)
        length[~good] = np.nan
        if np.isfinite(length).sum() < 3:
            continue  # this bone is never reliably seen; don't hold it against frames
        med = np.nanmedian(length)
        if not np.isfinite(med) or med <= 1e-6:
            continue
        dev = np.abs(length - med) / med
        measurable += good
        ok_counts += good & (dev <= tol)
        gross |= good & (dev > gross_tol)

    with np.errstate(invalid="ignore", divide="ignore"):
        plausibility = np.where(measurable > 0, ok_counts / measurable, 1.0)
    plausibility = np.clip(plausibility, 0.0, 1.0)
    mean = float(np.mean(plausibility)) if T else 1.0
    suspect = [int(i) for i in np.where((plausibility < suspect_below) | gross)[0]]
    return TrackingQuality(plausibility, mean, suspect)


@dataclass
class ViewClassification:
    view: str            # sagittal | frontal | oblique
    frontal_ratio: float  # (shoulder+hip horizontal spread) / stature
    certainty: str        # High | Medium | Low — how clear-cut the call is


def classify_view(kpts: np.ndarray) -> ViewClassification:
    """Classify the filming plane from left/right shoulder & hip horizontal
    separation relative to stature."""
    kpts = np.asarray(kpts, dtype=np.float64)
    body_h = estimate_body_height_px(kpts)
    if not np.isfinite(body_h) or body_h <= 0:
        return ViewClassification(OBLIQUE, float("nan"), "Low")

    def sep(a: str, b: str) -> np.ndarray:
        pa, pb = kpts[:, KP[a]], kpts[:, KP[b]]
        good = (pa[:, 2] >= MIN_CONF) & (pb[:, 2] >= MIN_CONF)
        s = np.abs(pa[:, 0] - pb[:, 0])
        s[~good] = np.nan
        return s

    shoulder = np.nanmedian(sep("l_shoulder", "r_shoulder"))
    hip = np.nanmedian(sep("l_hip", "r_hip"))
    spread = np.nanmean([shoulder, hip])
    ratio = float(spread / body_h) if np.isfinite(spread) else float("nan")
    if not np.isfinite(ratio):
        return ViewClassification(OBLIQUE, float("nan"), "Low")

    # Sagittal ≈ shoulders stacked (small spread); frontal ≈ a shoulder-width
    # apart (~0.2 of stature). Certainty is high away from the boundaries.
    if ratio < 0.10:
        view, edge = SAGITTAL, 0.10 - ratio
    elif ratio > 0.18:
        view, edge = FRONTAL, ratio - 0.18
    else:
        view, edge = OBLIQUE, min(ratio - 0.10, 0.18 - ratio)
    certainty = "High" if edge > 0.05 else "Medium" if edge > 0.02 else "Low"
    return ViewClassification(view, ratio, certainty)


# Which metric families are meaningful in which plane. Sprint sagittal angles
# (trunk/knee/thigh, contact/flight timing) need a side view; the jump valgus
# proxy needs a front view; distances/cadence are plane-robust.
def sagittal_metric_valid(view: str) -> bool:
    return view in (SAGITTAL, OBLIQUE)


def frontal_metric_valid(view: str) -> bool:
    return view in (FRONTAL, OBLIQUE)
