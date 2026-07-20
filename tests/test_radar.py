import numpy as np
import pytest

from athletic_analysis.core.metrics.sprint import SprintMetrics
from athletic_analysis.core.radar import (RADAR_AXES, band_score,
                                          compute_sprint_radar,
                                          symmetry_score)
from tests.conftest import make_sequence
from tests.test_coaching import FPS, _sprint_setup, _step


def _axis(radar, name):
    return next(a for a in radar.axes if a.name == name)


# --- band_score ---------------------------------------------------------------

def test_band_score_inside_band_is_100():
    assert band_score(0.12, 0.095, 0.15, 0.04) == 100.0
    assert band_score(0.095, 0.095, 0.15, 0.04) == 100.0  # at edge


def test_band_score_minor_major_anchors():
    # d == tol -> 60 (minor/major boundary), d == 3*tol -> 0.
    assert band_score(0.19, 0.095, 0.15, 0.04) == pytest.approx(60.0)
    assert band_score(0.27, 0.095, 0.15, 0.04) == pytest.approx(0.0, abs=1e-9)
    assert band_score(0.50, 0.095, 0.15, 0.04) == 0.0  # clamped


def test_band_score_low_side_and_nan():
    assert band_score(0.055, 0.095, 0.15, 0.04) == pytest.approx(60.0)
    assert np.isnan(band_score(float("nan"), 0.0, 1.0, 0.1))


# --- axes ---------------------------------------------------------------------

def test_good_form_scores_high_on_all_axes():
    kpts, metrics, vel = _sprint_setup()
    radar = compute_sprint_radar(kpts, metrics, vel, FPS)
    assert radar is not None and len(radar.axes) == len(RADAR_AXES)
    for axis in radar.axes:
        assert np.isfinite(axis.score) and axis.score >= 95, \
            (axis.name, axis.score, axis.detail)
    assert radar.overall >= 95


def test_long_contact_only_lowers_stiffness():
    kpts, metrics, vel = _sprint_setup(contact=0.30)
    radar = compute_sprint_radar(kpts, metrics, vel, FPS)
    assert _axis(radar, "Stiffness / contact").score < 60
    assert _axis(radar, "Posture / trunk").score >= 95
    assert _axis(radar, "Front-side mechanics").score >= 95


def test_level_tiering_rescoring():
    kpts, metrics, vel = _sprint_setup(contact=0.13)
    dev = compute_sprint_radar(kpts, metrics, vel, FPS, "developmental")
    elite = compute_sprint_radar(kpts, metrics, vel, FPS, "elite")
    assert (_axis(elite, "Stiffness / contact").score
            < _axis(dev, "Stiffness / contact").score)


def test_asymmetric_steps_lower_rhythm():
    kpts, metrics, vel = _sprint_setup()
    sym_rhythm = _axis(compute_sprint_radar(kpts, metrics, vel, FPS),
                       "Rhythm").score
    # Same cadence, but left steps much slower than right.
    asym = SprintMetrics()
    for i, step in enumerate(metrics.steps):
        contact = 0.14 if step.side == "left" else 0.09
        asym.steps.append(_step(step.strike_frame, step.side, contact=contact))
    asym_rhythm = _axis(compute_sprint_radar(kpts, asym, vel, FPS),
                        "Rhythm").score
    assert asym_rhythm < sym_rhythm


def test_symmetry_score_direct():
    even = [_step(25 * i, "left" if i % 2 == 0 else "right") for i in range(8)]
    assert symmetry_score(even) == 100.0
    skew = [_step(25 * i, "left" if i % 2 == 0 else "right",
                  step_time=0.20 if i % 2 == 0 else 0.28) for i in range(8)]
    assert symmetry_score(skew) < 60.0
    assert np.isnan(symmetry_score([]))


def test_drive_only_clip_leaves_maxv_axes_nan():
    # All steps classified as drive: rising speed, strikes early where the
    # speed ratio is < 0.70.
    kpts, metrics, _vel = _sprint_setup(trunk=40.0, contact=0.18)
    vel = {"run_speed": np.linspace(100, 1000, 300)}
    radar = compute_sprint_radar(kpts, metrics, vel, FPS)
    front = _axis(radar, "Front-side mechanics")  # thigh: max velocity only
    assert np.isnan(front.score) and front.n_steps == 0
    assert np.isfinite(radar.overall)  # other axes still count
    assert np.isfinite(_axis(radar, "Posture / trunk").score)


def test_no_steps_returns_none():
    kpts = make_sequence(50)
    assert compute_sprint_radar(kpts, SprintMetrics(), {}, FPS) is None
    assert compute_sprint_radar(kpts, None, {}, FPS) is None


def test_session_recompute_populates_radar(tmp_path):
    from athletic_analysis.core.session import AnalysisSession
    kpts, _metrics, _vel = _sprint_setup()
    session = AnalysisSession(video_path=str(tmp_path / "clip.mp4"), fps=FPS,
                              keypoints_raw=kpts.astype(np.float32))
    session.recompute()
    # Synthetic standing pose may not produce steps; the field must at least
    # be set without error, and be a SprintRadar when steps exist.
    if session.sprint_metrics and session.sprint_metrics.steps:
        assert session.sprint_radar is not None
    else:
        assert session.sprint_radar is None
