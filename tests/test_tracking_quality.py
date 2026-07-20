"""Tests for the anti-jitter / anti-ghost-detection fixes."""

import numpy as np

from athletic_analysis.core.filtering import remove_spikes, smooth_keypoints
from athletic_analysis.core.pose.rtmpose_backend import select_person
from athletic_analysis.core.velocity import compute_velocities, rolling_nanmean
from tests.conftest import make_sequence


# --- person selection ---------------------------------------------------------

def _people(*mean_scores_and_centers):
    """Build (keypoints, scores) for fake people at given (score, (x, y))."""
    kpts, scores = [], []
    for score, (x, y) in mean_scores_and_centers:
        kpts.append(np.full((26, 2), (x, y), dtype=np.float64))
        scores.append(np.full(26, score))
    return np.stack(kpts), np.stack(scores)


def test_no_credible_person_returns_none():
    kpts, scores = _people((0.15, (100, 100)), (0.2, (500, 300)))
    assert select_person(kpts, scores, None, min_conf=0.35, img_diag=1000) is None


def test_highest_score_wins_without_history():
    kpts, scores = _people((0.5, (100, 100)), (0.9, (500, 300)))
    assert select_person(kpts, scores, None, min_conf=0.35, img_diag=1000) == 1


def test_sticks_with_tracked_athlete_over_new_detection():
    # Slightly higher-scoring person far away must not steal the track.
    kpts, scores = _people((0.75, (100, 100)), (0.85, (900, 500)))
    last = np.array([105.0, 102.0])
    assert select_person(kpts, scores, last, min_conf=0.35, img_diag=1000) == 0


# --- spike removal ------------------------------------------------------------

def test_remove_spikes_kills_single_frame_jump():
    x = np.linspace(0, 100, 60)
    y = np.full(60, 50.0)
    x_spiked = x.copy()
    x_spiked[30] += 400.0  # detector briefly jumped to a background object
    fixed_x, _fixed_y = remove_spikes(x_spiked, y)
    assert abs(fixed_x[30] - x[30]) < 5.0
    # Non-spike samples untouched.
    assert np.allclose(fixed_x[:29], x[:29])


def test_remove_spikes_leaves_clean_signal_alone():
    x = np.linspace(0, 100, 60)
    y = 50 + 10 * np.sin(np.linspace(0, 6, 60))
    fixed_x, fixed_y = remove_spikes(x, y)
    assert np.allclose(fixed_x, x)
    assert np.allclose(fixed_y, y)


def test_smooth_keypoints_absorbs_spike():
    kpts = make_sequence(80)
    kpts[:, 0, 0] = np.linspace(0, 200, 80)
    kpts[40, 0, 0] = 5000.0  # huge one-frame outlier, full confidence
    out = smooth_keypoints(kpts, fps=30.0)
    assert abs(out[40, 0, 0] - np.linspace(0, 200, 80)[40]) < 15


# --- run speed ----------------------------------------------------------------

def test_rolling_nanmean_ignores_nans():
    x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    out = rolling_nanmean(x, 3)
    assert np.isclose(out[1], 2.0)  # mean of 1 and 3


def test_run_speed_not_inflated_by_vertical_bounce():
    from athletic_analysis.core.pose.skeleton import KP
    kpts = make_sequence(200)
    fps = 100.0
    kpts[:, :, 0] += np.arange(200)[:, None] * 8.0  # constant 800 px/s forward
    # Add a strong stride bounce to the hip (does not change forward speed).
    kpts[:, KP["hip_center"], 1] += 30 * np.sin(np.arange(200) * 0.6)
    vel, _unit = compute_velocities(kpts, fps)
    mid = slice(30, 170)
    # hip_speed (instantaneous, includes bounce) should exceed run_speed,
    # and run_speed should match the true forward speed (~1.5 BH/s).
    assert np.nanmean(vel["hip_speed"][mid]) > np.nanmean(vel["run_speed"][mid])
    forward = np.nanmean(np.abs(vel["hip_vx"][mid]))
    assert np.isclose(np.nanmean(vel["run_speed"][mid]), forward, rtol=0.05)
