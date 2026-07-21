import numpy as np

from athletic_analysis.core.pose.skeleton import KP
from athletic_analysis.core.quality import (FRONTAL, SAGITTAL, classify_view,
                                            frontal_metric_valid,
                                            sagittal_metric_valid,
                                            tracking_quality)
from tests.conftest import make_sequence, standing_pose


def test_tracking_quality_clean_track_is_fully_plausible():
    kpts = make_sequence(60)  # rigid standing pose repeated: constant bones
    tq = tracking_quality(kpts)
    assert tq.mean_plausibility > 0.98
    assert tq.suspect_frames == []


def test_tracking_quality_flags_a_leg_swap():
    kpts = make_sequence(60)
    # Simulate a one-frame L/R swap: knee/ankle jump across the body, which
    # blows up the thigh/shank lengths on that frame.
    bad = 30
    for name in ("l_knee", "l_ankle"):
        kpts[bad, KP[name], 0] += 400  # yank the left leg far to the right
    tq = tracking_quality(kpts)
    assert bad in tq.suspect_frames
    assert tq.plausibility[bad] < tq.plausibility[0]


def test_classify_view_sagittal_when_shoulders_stacked():
    kpts = make_sequence(30)
    # Collapse left/right onto the same x (a pure side view projects them there).
    for pair in (("l_shoulder", "r_shoulder"), ("l_hip", "r_hip")):
        for name in pair:
            kpts[:, KP[name], 0] = 300.0
    v = classify_view(kpts)
    assert v.view == SAGITTAL
    assert sagittal_metric_valid(v.view) and not frontal_metric_valid(v.view)


def test_classify_view_frontal_when_shoulders_spread():
    kpts = make_sequence(30)
    # Spread shoulders/hips ~a shoulder-width apart horizontally (front view).
    body_span = 130.0
    kpts[:, KP["l_shoulder"], 0] = 300 - body_span / 2
    kpts[:, KP["r_shoulder"], 0] = 300 + body_span / 2
    kpts[:, KP["l_hip"], 0] = 300 - body_span / 2
    kpts[:, KP["r_hip"], 0] = 300 + body_span / 2
    v = classify_view(kpts)
    assert v.view == FRONTAL
    assert frontal_metric_valid(v.view) and not sagittal_metric_valid(v.view)


def test_view_gating_helpers_agree_on_oblique():
    # Oblique keeps both nominally valid (degraded, not invalid).
    assert sagittal_metric_valid("oblique")
    assert frontal_metric_valid("oblique")
