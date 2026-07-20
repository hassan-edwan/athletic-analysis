"""Lightweight person detector (rtmlib YOLOX) shared by assessment and reframe.

Returns every person's bounding box + score per frame, unlike the pose backend
which collapses to a single primary person. Loads the same YOLOX weights the
pose pipeline already caches — no extra download.
"""

from __future__ import annotations

import numpy as np


class PersonDetector:
    def __init__(self, mode: str = "balanced"):
        from rtmlib import YOLOX

        # rtmlib caches this model under ~/.cache/rtmlib; BodyWithFeet uses the
        # same detector, so this reuses already-downloaded weights.
        self._model = YOLOX(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
            model_input_size=(640, 640),
            backend="onnxruntime", device="cpu")

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return an (N, 4) array of person boxes [x1, y1, x2, y2]."""
        boxes = self._model(frame_bgr)
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def largest_box(boxes: np.ndarray) -> np.ndarray | None:
    """The tallest box — a robust 'primary athlete' pick for single-subject clips."""
    if boxes is None or len(boxes) == 0:
        return None
    heights = boxes[:, 3] - boxes[:, 1]
    return boxes[int(np.argmax(heights))]


def box_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def select_tracked_box(boxes: np.ndarray, last_center: np.ndarray | None
                       ) -> np.ndarray | None:
    """Pick the athlete across frames with temporal consistency: the tallest
    box initially, then the box nearest the previous pick. Prevents the crop
    from jumping between people/panels when another detection is momentarily
    taller (the multi-panel / bystander failure mode)."""
    if boxes is None or len(boxes) == 0:
        return None
    if last_center is None:
        return largest_box(boxes)
    centers = np.stack([box_center(b) for b in boxes])
    dist = np.linalg.norm(centers - last_center, axis=1)
    return boxes[int(np.argmin(dist))]
