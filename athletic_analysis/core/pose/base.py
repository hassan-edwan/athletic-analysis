"""Abstract pose-estimation backend so the model can be swapped (RTMPose, MediaPipe, ...)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PoseBackend(ABC):
    """Estimates a single (primary) person's 2D keypoints per frame.

    Output convention everywhere in this app: float array of shape
    (num_keypoints, 3) = (x_px, y_px, confidence in [0, 1]).
    """

    num_keypoints: int = 26  # Halpe-26 layout (see skeleton.py)

    @abstractmethod
    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return (num_keypoints, 3) for the primary person; zeros if nobody found."""

    def empty(self) -> np.ndarray:
        return np.zeros((self.num_keypoints, 3), dtype=np.float32)
