import numpy as np

from athletic_analysis.core.confidence import (clip_quality, detection_factor,
                                               metric_confidence,
                                               sample_factor, temporal_factor)
from athletic_analysis.core.pose.skeleton import KP
from tests.conftest import make_sequence


def test_temporal_factor_fps_matters_for_fixed_contact():
    # A 90 ms ground contact spans 2.7 frames at 30 fps vs 21.6 at 240 fps.
    low = temporal_factor(0.09 * 30)
    high = temporal_factor(0.09 * 240)
    assert low < 0.5 < high
    assert high == 1.0


def test_temporal_factor_bounds():
    assert temporal_factor(0) == 0.4
    assert temporal_factor(float("nan")) == 0.4
    assert temporal_factor(100) == 1.0


def test_detection_factor_drops_when_joint_lost():
    kpts = make_sequence(20)
    frames = list(range(20))
    full = detection_factor(kpts, ["l_knee", "r_knee"], frames)
    assert full > 0.95
    kpts[:, KP["l_knee"], 2] = 0.0
    lost = detection_factor(kpts, ["l_knee", "r_knee"], frames)
    assert lost < full


def test_sample_factor_grows_with_steps():
    assert sample_factor(1) < sample_factor(4)
    assert sample_factor(4) == 1.0
    assert sample_factor(0) == 0.4


def test_metric_confidence_names_worst_limiter():
    # Good detection but only 3 frames of contact at low fps => frame-rate limited.
    c = metric_confidence(detection=0.9, frames_spanned=3, n_samples=5)
    assert c.level in ("Low", "Medium")
    assert c.limiter == "frame rate"


def test_uncalibrated_distance_capped_at_medium():
    c = metric_confidence(detection=0.95, uncalibrated_distance=True)
    assert c.level == "Medium"
    assert c.limiter == "not calibrated"


def test_calibrated_distance_can_be_high():
    c = metric_confidence(detection=0.95, uncalibrated_distance=False)
    assert c.level == "High"


def test_clip_quality_flags_low_fps_and_calibration():
    kpts = make_sequence(60)
    q = clip_quality(kpts, fps=30.0, calibrated=False)
    assert q.detection_rate > 0.95
    assert not q.fps_adequate
    assert any("fps" in n for n in q.notes)
    assert any("uncalibrated" in n for n in q.notes)


def test_clip_quality_high_when_good():
    kpts = make_sequence(120)
    q = clip_quality(kpts, fps=120.0, calibrated=True)
    assert q.fps_adequate
    assert q.level == "High"


def test_clip_quality_detection_rate_reflects_lost_frames():
    kpts = make_sequence(100)
    kpts[40:60, KP["hip_center"], 2] = 0.0  # person lost for 20 frames
    q = clip_quality(kpts, fps=60.0, calibrated=True)
    assert np.isclose(q.detection_rate, 0.8, atol=0.02)
