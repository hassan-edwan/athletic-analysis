"""Per-frame velocity series from smoothed keypoint trajectories.

Units: m/s when calibrated, otherwise body-heights per second (BH/s).
Vertical velocity is physics-style: positive = upward (image y grows down).
"""

from __future__ import annotations

import numpy as np

from athletic_analysis.core.angles import MIN_CONF, estimate_body_height_px
from athletic_analysis.core.calibration import Calibration
from athletic_analysis.core.pose.skeleton import KP


def rolling_nanmean(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average that ignores NaNs (NaN where the window is empty)."""
    n = len(x)
    if n == 0 or window <= 1:
        return x.copy()
    valid = np.isfinite(x)
    csum = np.cumsum(np.where(valid, x, 0.0))
    ccnt = np.cumsum(valid.astype(np.float64))
    half = window // 2
    idx = np.arange(n)
    lo = np.maximum(idx - half, 0)
    hi = np.minimum(idx + half, n - 1)
    sums = csum[hi] - np.where(lo > 0, csum[lo - 1], 0.0)
    cnts = ccnt[hi] - np.where(lo > 0, ccnt[lo - 1], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sums / cnts
    out[cnts == 0] = np.nan
    return out


def compute_velocities(kpts: np.ndarray, fps: float,
                       calib: Calibration | None = None
                       ) -> tuple[dict[str, np.ndarray], str]:
    """Returns ({hip_vx, hip_vy, hip_speed, run_speed}, unit). Arrays are (T,),
    NaN where the hip is low-confidence.

    `run_speed` is what a coach means by "how fast are they running here":
    horizontal velocity magnitude averaged over ~0.4 s, so it isn't inflated
    by the vertical bounce of each stride or by frame-to-frame noise."""
    kpts = np.asarray(kpts, dtype=np.float64)
    if calib is not None:
        scale, unit = calib.meters_per_pixel, "m/s"
    else:
        body_h = estimate_body_height_px(kpts)
        if np.isfinite(body_h) and body_h > 0:
            scale, unit = 1.0 / body_h, "BH/s"
        else:
            scale, unit = np.nan, "?/s"

    hip = kpts[:, KP["hip_center"]]
    if len(kpts) < 2:
        empty = np.full(len(kpts), np.nan)
        return {"hip_vx": empty, "hip_vy": empty.copy(),
                "hip_speed": empty.copy(), "run_speed": empty.copy()}, unit
    vx = np.gradient(hip[:, 0]) * fps * scale
    vy = -np.gradient(hip[:, 1]) * fps * scale  # up-positive
    bad = hip[:, 2] < MIN_CONF
    vx[bad] = np.nan
    vy[bad] = np.nan
    run = rolling_nanmean(np.abs(vx), max(3, round(0.4 * fps)))
    run[bad] = np.nan
    return {"hip_vx": vx, "hip_vy": vy, "hip_speed": np.hypot(vx, vy),
            "run_speed": run}, unit
