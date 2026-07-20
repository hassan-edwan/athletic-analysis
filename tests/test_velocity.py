import numpy as np

from athletic_analysis.core.calibration import Calibration
from athletic_analysis.core.events import JumpPhases
from athletic_analysis.core.metrics.jump import compute_jump_metrics
from athletic_analysis.core.angles import compute_angles
from athletic_analysis.core.pose.skeleton import KP
from athletic_analysis.core.velocity import compute_velocities
from tests.conftest import make_sequence

FPS = 100.0


def test_constant_horizontal_speed_calibrated():
    kpts = make_sequence(100)
    kpts[:, :, 0] += np.arange(100)[:, None] * 8.0  # 8 px/frame -> 800 px/s
    calib = Calibration(meters_per_pixel=0.005)  # 800 px/s -> 4.0 m/s
    vel, unit = compute_velocities(kpts, FPS, calib)
    assert unit == "m/s"
    assert np.allclose(vel["hip_vx"][5:-5], 4.0, atol=0.05)
    assert np.allclose(vel["hip_vy"][5:-5], 0.0, atol=0.05)
    assert np.allclose(vel["hip_speed"][5:-5], 4.0, atol=0.05)


def test_vertical_velocity_up_positive():
    kpts = make_sequence(100)
    kpts[:, KP["hip_center"], 1] -= np.arange(100) * 2.0  # rising (y decreases)
    vel, unit = compute_velocities(kpts, FPS)
    assert unit == "BH/s"
    assert np.nanmean(vel["hip_vy"][5:-5]) > 0


def test_uncalibrated_uses_body_heights():
    kpts = make_sequence(100)
    kpts[:, :, 0] += np.arange(100)[:, None] * 8.0
    vel, unit = compute_velocities(kpts, FPS)
    assert unit == "BH/s"
    # 800 px/s over a ~530 px body -> ~1.5 BH/s
    assert 1.0 < np.nanmean(vel["hip_speed"][5:-5]) < 2.0


def test_low_confidence_hip_gives_nan():
    kpts = make_sequence(50)
    kpts[10:20, KP["hip_center"], 2] = 0.0
    vel, _unit = compute_velocities(kpts, FPS)
    assert np.isnan(vel["hip_vx"][12:18]).all()


def test_takeoff_velocity_from_flight_time():
    kpts = make_sequence(300)
    jump = JumpPhases(takeoff_frame=150, landing_frame=190,
                      lowest_hip_frame=140, baseline_hip_y=330.0)
    m = compute_jump_metrics(kpts, compute_angles(kpts), jump, FPS)
    # v0 = g * t / 2 with t = 0.4 s -> 1.962 m/s
    assert np.isclose(m.takeoff_velocity_m_s, 9.81 * 0.4 / 2, atol=1e-3)
