import numpy as np

from athletic_analysis.core.filtering import lowpass, smooth_keypoints
from tests.conftest import make_sequence


def test_lowpass_reduces_noise():
    rng = np.random.default_rng(0)
    fps = 100.0
    t = np.arange(300) / fps
    clean = 100 * np.sin(2 * np.pi * 1.5 * t)
    noisy = clean + rng.normal(0, 8, len(t))
    filtered = lowpass(noisy, fps, cutoff_hz=6.0)
    assert np.std(filtered - clean) < np.std(noisy - clean) / 2


def test_short_signal_does_not_crash():
    out = lowpass(np.array([1.0, 2.0, 3.0]), fps=30.0)
    assert len(out) == 3


def test_gap_interpolation():
    kpts = make_sequence(60)
    kpts[:, 0, 0] = np.linspace(0, 590, 60)  # nose moves linearly
    kpts[20:30, 0, 2] = 0.0  # drop confidence in the middle
    kpts[20:30, 0, 0] = 9999.0  # and corrupt the coordinates there
    out = smooth_keypoints(kpts, fps=30.0)
    # Interpolation + smoothing should stay near the linear trajectory.
    expected = np.linspace(0, 590, 60)[20:30]
    assert np.abs(out[20:30, 0, 0] - expected).max() < 20
