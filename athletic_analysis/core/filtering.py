"""Keypoint trajectory conditioning: gap interpolation + zero-lag Butterworth smoothing."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt


def _fill_gaps(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly interpolate over invalid samples; hold ends."""
    if valid.all():
        return values
    if not valid.any():
        return values
    idx = np.arange(len(values))
    out = values.copy()
    out[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
    return out


def remove_spikes(x: np.ndarray, y: np.ndarray,
                  factor: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Drop single-frame position spikes (detector briefly latching onto
    something else) and re-interpolate them. A sample is a spike when both the
    jump into it and out of it far exceed the median frame-to-frame movement."""
    d = np.hypot(np.diff(x), np.diff(y))
    moving = d[d > 1e-9]
    if len(moving) < 4:
        return x, y
    thresh = factor * float(np.median(moving))
    if thresh <= 1e-9:
        return x, y
    jump_in = np.concatenate(([0.0], d))
    jump_out = np.concatenate((d, [0.0]))
    bad = np.zeros(len(x), dtype=bool)
    bad[1:-1] = (jump_in[1:-1] > thresh) & (jump_out[1:-1] > thresh)
    if not bad.any() or bad.all():
        return x, y
    idx = np.arange(len(x))
    x, y = x.copy(), y.copy()
    x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    y[bad] = np.interp(idx[bad], idx[~bad], y[~bad])
    return x, y


def lowpass(signal: np.ndarray, fps: float, cutoff_hz: float = 6.0, order: int = 4) -> np.ndarray:
    """Zero-lag Butterworth low-pass (the biomechanics standard for kinematics)."""
    nyq = fps / 2.0
    cutoff = min(cutoff_hz, nyq * 0.95)
    b, a = butter(order, cutoff / nyq)
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(signal) <= padlen:
        # Too short for filtfilt: fall back to a small moving average.
        k = max(1, min(5, len(signal)))
        kernel = np.ones(k) / k
        return np.convolve(signal, kernel, mode="same")
    return filtfilt(b, a, signal)


def smooth_keypoints(kpts: np.ndarray, fps: float, cutoff_hz: float = 6.0,
                     min_conf: float = 0.3) -> np.ndarray:
    """Condition raw per-frame keypoints.

    kpts: (T, K, 3) raw estimates. Returns same shape: x/y interpolated across
    low-confidence gaps then low-pass filtered; confidence channel untouched.
    """
    kpts = np.asarray(kpts, dtype=np.float64)
    out = kpts.copy()
    T, K, _ = kpts.shape
    if T < 3:
        return out
    for k in range(K):
        valid = kpts[:, k, 2] >= min_conf
        if valid.sum() < 2:
            continue
        x = _fill_gaps(kpts[:, k, 0], valid)
        y = _fill_gaps(kpts[:, k, 1], valid)
        x, y = remove_spikes(x, y)
        out[:, k, 0] = lowpass(x, fps, cutoff_hz)
        out[:, k, 1] = lowpass(y, fps, cutoff_hz)
        # Median-filter confidence so bones don't flicker in and out of the
        # overlay when scores hover around the draw threshold.
        if T >= 5:
            out[:, k, 2] = median_filter(kpts[:, k, 2], size=5)
    return out
