"""Video preprocessing transforms that rescue bad footage for pose analysis.

All transforms operate on real pixels (no fabricated frames). Geometry-changing
steps carry a coordinate mapping so keypoints estimated on the processed frame
can be placed back in display-space coordinates.

Pipeline order (applied by AnalysisWorker): rotation (at decode, VideoSource) →
reframe (crop+upscale the athlete) → enhance / deinterlace (pixel-only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from athletic_analysis.core.angles import MIN_CONF

# --- simple pixel transforms --------------------------------------------------


def keypoints_bbox(kpts: np.ndarray, min_conf: float = MIN_CONF
                   ) -> np.ndarray | None:
    """[x1, y1, x2, y2] enclosing the confidently-tracked keypoints in one
    frame's (26, 3) array, or None when fewer than two are confident (too
    few to form a sane box). Used to derive a ReframeTracker trajectory
    straight from pose output, without a separate detector pass."""
    good = kpts[:, 2] >= min_conf
    if int(good.sum()) < 2:
        return None
    pts = kpts[good, :2]
    return np.array([pts[:, 0].min(), pts[:, 1].min(),
                     pts[:, 0].max(), pts[:, 1].max()])


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate 0/90/180/270 degrees clockwise."""
    d = degrees % 360
    if d == 0:
        return frame
    if d == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if d == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if d == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("rotation must be a multiple of 90")


def enhance_contrast(frame: np.ndarray, clip_limit: float = 2.5,
                     gamma: float = 1.0) -> np.ndarray:
    """CLAHE on the L channel (local contrast) plus optional gamma. Rescues
    dark / washed-out footage so the detector finds the athlete."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    if abs(gamma - 1.0) > 1e-3:
        inv = 1.0 / max(gamma, 1e-3)
        table = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
        out = cv2.LUT(out, table)
    return out


def comb_metric(frame: np.ndarray) -> float:
    """Interlacing 'comb' strength: mean abs difference between a row and the
    average of its vertical neighbours, normalized. High on interlaced motion."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    neighbour_avg = (g[:-2] + g[2:]) / 2.0
    comb = np.abs(g[1:-1] - neighbour_avg)
    return float(comb.mean() / (g.std() + 1e-6))


def deinterlace(frame: np.ndarray) -> np.ndarray:
    """Bob deinterlace: keep the even field and interpolate the missing rows.
    Removes comb artifacts on moving limbs (geometry-preserving)."""
    h = frame.shape[0]
    even = frame[0::2]
    return cv2.resize(even, (frame.shape[1], h), interpolation=cv2.INTER_LINEAR)


# --- athlete-tracking reframe -------------------------------------------------


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < 2:
        return x
    k = np.ones(window) / window
    return np.convolve(x, k, mode="same")


@dataclass
class ReframeTracker:
    """Turns a per-frame athlete-bbox trajectory into stable, upscaled crops.

    `boxes[i]` is [x1, y1, x2, y2] for frame i, or None where undetected.
    Missing boxes are interpolated; the trajectory is smoothed; each crop is
    padded, aspect-locked (3:4), and clamped to the frame."""

    boxes: list[np.ndarray | None]
    frame_w: int
    frame_h: int
    pad: float = 0.35          # fraction of bbox size added around the athlete
    aspect: float = 3 / 4      # crop width:height (athletes are taller than wide)
    target_h: int = 384        # upscale small crops so height reaches this
    smooth_window: int = 7
    _crops: list[tuple[int, int, int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.boxes)
        cx = np.full(n, np.nan)
        cy = np.full(n, np.nan)
        bw = np.full(n, np.nan)
        bh = np.full(n, np.nan)
        for i, box in enumerate(self.boxes):
            if box is None:
                continue
            x1, y1, x2, y2 = box
            cx[i] = (x1 + x2) / 2
            cy[i] = (y1 + y2) / 2
            bw[i] = x2 - x1
            bh[i] = y2 - y1
        if not np.isfinite(cx).any():
            # No detections at all: crop = whole frame (identity).
            self._crops = [(0, 0, self.frame_w, self.frame_h)] * n
            return
        idx = np.arange(n)
        for arr in (cx, cy, bw, bh):
            good = np.isfinite(arr)
            arr[~good] = np.interp(idx[~good], idx[good], arr[good])
        cx = _moving_average(cx, self.smooth_window)
        cy = _moving_average(cy, self.smooth_window)
        bw = _moving_average(bw, self.smooth_window)
        bh = _moving_average(bh, self.smooth_window)

        for i in range(n):
            self._crops.append(self._crop_rect(cx[i], cy[i], bw[i], bh[i]))

    def _crop_rect(self, cx: float, cy: float, bw: float, bh: float
                   ) -> tuple[int, int, int, int]:
        # Pad, then lock to the target aspect (choose the larger dimension).
        w = bw * (1 + 2 * self.pad)
        h = bh * (1 + 2 * self.pad)
        if w / h > self.aspect:
            h = w / self.aspect
        else:
            w = h * self.aspect
        w = min(w, self.frame_w)
        h = min(h, self.frame_h)
        x0 = int(round(min(max(cx - w / 2, 0), self.frame_w - w)))
        y0 = int(round(min(max(cy - h / 2, 0), self.frame_h - h)))
        return (x0, y0, int(round(w)), int(round(h)))

    def crop_and_map(self, frame: np.ndarray, idx: int
                     ) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        """Return (processed_crop, to_original) for frame `idx`. `to_original`
        maps an (K, 2) or (K, 3) keypoint array from crop space to full-frame
        coordinates."""
        x0, y0, w, h = self._crops[idx] if idx < len(self._crops) \
            else (0, 0, self.frame_w, self.frame_h)
        crop = frame[y0:y0 + h, x0:x0 + w]
        if crop.size == 0:
            return frame, lambda pts: pts
        scale = float(np.clip(self.target_h / max(h, 1), 1.0, 4.0))
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_CUBIC)
        sx, sy = w / tw, h / th

        def to_original(pts: np.ndarray) -> np.ndarray:
            pts = np.asarray(pts, dtype=np.float64).copy()
            pts[..., 0] = pts[..., 0] * sx + x0
            pts[..., 1] = pts[..., 1] * sy + y0
            return pts

        return resized, to_original


# --- preprocessor bundle ------------------------------------------------------

TRANSFORM_NAMES = ("reframe", "enhance", "deinterlace")


@dataclass
class Preprocessor:
    """Bundle of enabled transforms for one analysis run. `rotation` is handled
    upstream by VideoSource; kept here only for the record of what was applied."""

    rotation: int = 0
    enhance: bool = False
    deinterlace: bool = False
    reframe: bool = False
    tracker: ReframeTracker | None = None

    def needs_detection_pass(self) -> bool:
        return self.reframe

    def process(self, frame: np.ndarray, idx: int
                ) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        """Apply pixel transforms; return (frame, to_original keypoint map)."""
        to_original: Callable[[np.ndarray], np.ndarray] = lambda pts: pts
        # Deinterlace the full frame first (geometry-preserving), so the reframe
        # crop and pose run on clean rows.
        if self.deinterlace:
            frame = deinterlace(frame)
        if self.reframe and self.tracker is not None:
            frame, to_original = self.tracker.crop_and_map(frame, idx)
        if self.enhance:
            frame = enhance_contrast(frame)
        return frame, to_original

    def applied(self) -> list[str]:
        names = []
        if self.rotation:
            names.append(f"rotate{self.rotation}")
        for name in TRANSFORM_NAMES:
            if getattr(self, name):
                names.append(name)
        return names
