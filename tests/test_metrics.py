import numpy as np

from athletic_analysis.core.angles import compute_angles
from athletic_analysis.core.calibration import Calibration
from athletic_analysis.core.events import GaitEvent, JumpPhases
from athletic_analysis.core.metrics.jump import compute_jump_metrics
from athletic_analysis.core.metrics.sprint import compute_sprint_metrics
from athletic_analysis.core.pose.skeleton import KP
from tests.conftest import make_sequence

FPS = 100.0


def test_jump_height_from_flight_time():
    kpts = make_sequence(300)
    jump = JumpPhases(takeoff_frame=150, landing_frame=190,
                      lowest_hip_frame=140, baseline_hip_y=330.0)
    angles = compute_angles(kpts)
    m = compute_jump_metrics(kpts, angles, jump, FPS)
    # h = g * t^2 / 8 with t = 0.4 s -> 0.1962 m
    assert np.isclose(m.flight_time_s, 0.4)
    assert np.isclose(m.jump_height_flight_m, 9.81 * 0.16 / 8, atol=1e-3)


def test_jump_hip_rise_calibrated():
    kpts = make_sequence(300)
    # Hip rises 100 px above baseline during flight.
    kpts[150:190, KP["hip_center"], 1] = 230.0
    jump = JumpPhases(takeoff_frame=150, landing_frame=190,
                      lowest_hip_frame=140, baseline_hip_y=330.0)
    angles = compute_angles(kpts)
    calib = Calibration(meters_per_pixel=0.002)  # 100 px = 0.2 m
    m = compute_jump_metrics(kpts, angles, jump, FPS, calib)
    assert m.length_unit == "m"
    assert np.isclose(m.hip_rise, 0.2, atol=0.01)


def test_no_jump_returns_empty_metrics():
    kpts = make_sequence(50)
    m = compute_jump_metrics(kpts, compute_angles(kpts), None, FPS)
    assert m.takeoff_frame == -1
    assert np.isnan(m.flight_time_s)


def _make_events(step_frames: list[tuple[int, str]], contact_len: int) -> list[GaitEvent]:
    events = []
    for frame, side in step_frames:
        events.append(GaitEvent(frame=frame, side=side, kind="strike"))
        events.append(GaitEvent(frame=frame + contact_len, side=side, kind="toeoff"))
    events.sort(key=lambda ev: ev.frame)
    return events


def test_sprint_metrics_from_known_events():
    T = 300
    kpts = make_sequence(T)
    speed_px = 8.0  # px per frame, rightward
    kpts[:, :, 0] += np.arange(T)[:, None] * speed_px
    angles = compute_angles(kpts)
    # Steps every 25 frames (0.25 s -> 240 steps/min), 12-frame contacts.
    steps = [(25 * i, "left" if i % 2 == 0 else "right") for i in range(1, 10)]
    events = _make_events(steps, contact_len=12)
    m = compute_sprint_metrics(kpts, angles, events, FPS)
    assert np.isclose(m.cadence_spm, 240.0, atol=5)
    assert np.isclose(m.mean_contact_s, 0.12, atol=0.01)
    # Step length: 25 frames * 8 px = 200 px; in body-heights (~530 px) ~0.38.
    assert 0.25 < m.mean_step_length < 0.55
    assert m.length_unit == "BH"


def test_sprint_metrics_calibrated_step_length():
    T = 300
    kpts = make_sequence(T)
    kpts[:, :, 0] += np.arange(T)[:, None] * 8.0
    angles = compute_angles(kpts)
    steps = [(25 * i, "left" if i % 2 == 0 else "right") for i in range(1, 10)]
    events = _make_events(steps, contact_len=12)
    calib = Calibration(meters_per_pixel=0.005)  # 200 px = 1.0 m
    m = compute_sprint_metrics(kpts, angles, events, FPS, calib)
    assert m.length_unit == "m"
    assert np.isclose(m.mean_step_length, 1.0, atol=0.05)
    # 8 px/frame at 100 fps with 0.005 m/px -> 4 m/s, per step and overall.
    step_speeds = [s.step_speed for s in m.steps if np.isfinite(s.step_speed)]
    assert step_speeds and np.allclose(step_speeds, 4.0, atol=0.1)
    assert np.isclose(m.mean_speed, 4.0, atol=0.1)


def test_sprint_metrics_empty_events():
    kpts = make_sequence(50)
    m = compute_sprint_metrics(kpts, compute_angles(kpts), [], FPS)
    assert m.steps == []
    assert np.isnan(m.cadence_spm)
