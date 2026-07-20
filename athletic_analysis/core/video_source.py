"""Frame-accurate video reading with an LRU frame cache.

Random seeks via CAP_PROP_POS_FRAMES are unreliable on many codecs, so all
reads decode sequentially. Backward or long forward jumps do a coarse seek to
a point *before* the target and decode forward to it.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

# Decoding forward this many frames is cheaper than risking an inexact seek.
_SEEK_MARGIN = 8
_MAX_FORWARD_DECODE = 90


class VideoSource:
    def __init__(self, path: str | Path, cache_size: int = 240,
                 rotation: int = 0):
        self.path = str(path)
        self.rotation = rotation % 360
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise IOError(f"Could not open video: {self.path}")
        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.fps <= 0:
            self.fps = 30.0
        self.frame_count: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Reported dimensions reflect the rotated (display) orientation.
        if self.rotation in (90, 270):
            self.width, self.height = h, w
        else:
            self.width, self.height = w, h
        self._next_idx = 0  # index the next cap.read() will return
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    def _rotate(self, frame: np.ndarray) -> np.ndarray:
        if self.rotation == 0:
            return frame
        from athletic_analysis.core.preprocess import rotate_frame
        return rotate_frame(frame, self.rotation)

    def close(self) -> None:
        self._cap.release()
        self._cache.clear()

    def get_frame(self, idx: int) -> Optional[np.ndarray]:
        """Return BGR frame `idx`, or None past end of video."""
        if self.frame_count > 0:
            idx = max(0, min(idx, self.frame_count - 1))
        else:
            idx = max(0, idx)
        cached = self._cache.get(idx)
        if cached is not None:
            self._cache.move_to_end(idx)
            return cached
        if idx < self._next_idx or idx - self._next_idx > _MAX_FORWARD_DECODE:
            target = max(0, idx - _SEEK_MARGIN)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self._next_idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame = None
        while self._next_idx <= idx:
            ok, frame = self._cap.read()
            if not ok:
                return self._cache.get(idx)
            frame = self._rotate(frame)
            self._store(self._next_idx, frame)
            self._next_idx += 1
        return frame

    def _store(self, idx: int, frame: np.ndarray) -> None:
        self._cache[idx] = frame
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def iter_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Sequential full-video decode (own capture; safe alongside get_frame)."""
        cap = cv2.VideoCapture(self.path)
        try:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, self._rotate(frame)
                idx += 1
        finally:
            cap.release()

    def frame_to_time(self, idx: int) -> float:
        return idx / self.fps
