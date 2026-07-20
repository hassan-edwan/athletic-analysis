import numpy as np

from athletic_analysis.core.angles import (angle_3pt, compute_angles,
                                           estimate_body_height_px,
                                           travel_direction)
from athletic_analysis.core.pose.skeleton import KP
from tests.conftest import make_sequence


def test_angle_3pt_right_angle():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    assert np.isclose(angle_3pt(a, b, c), 90.0)


def test_angle_3pt_straight_line():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([2.0, 0.0])
    assert np.isclose(angle_3pt(a, b, c), 180.0)


def test_standing_knee_nearly_extended(pose_seq):
    angles = compute_angles(pose_seq)
    assert np.nanmean(angles["knee_l"]) > 170
    assert np.nanmean(angles["knee_r"]) > 170


def test_standing_trunk_vertical(pose_seq):
    angles = compute_angles(pose_seq)
    assert abs(np.nanmean(angles["trunk_lean"])) < 5


def test_forward_lean_positive_toward_travel():
    kpts = make_sequence(50)
    # Move rightward and lean shoulders 40 px to the right of the hips.
    kpts[:, :, 0] += np.arange(50)[:, None] * 5
    for name in ("l_shoulder", "r_shoulder", "neck"):
        kpts[:, KP[name], 0] += 40
    angles = compute_angles(kpts)
    assert np.nanmean(angles["trunk_lean"]) > 5


def test_low_confidence_gives_nan(pose_seq):
    kpts = pose_seq.copy()
    kpts[:, KP["l_knee"], 2] = 0.0
    angles = compute_angles(kpts)
    assert np.isnan(angles["knee_l"]).all()
    assert np.isfinite(angles["knee_r"]).all()


def test_travel_direction(pose_seq):
    rightward = pose_seq.copy()
    rightward[:, :, 0] += np.arange(len(rightward))[:, None] * 3
    assert travel_direction(rightward) == 1.0
    leftward = pose_seq.copy()
    leftward[:, :, 0] -= np.arange(len(leftward))[:, None] * 3
    assert travel_direction(leftward) == -1.0


def test_body_height_estimate_plausible(pose_seq):
    h = estimate_body_height_px(pose_seq)
    # Figure spans roughly 90..600 px -> stature estimate should be in that ballpark.
    assert 400 < h < 650
