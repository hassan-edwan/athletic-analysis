"""RTMPose (rtmlib) backend using the Halpe-26 body+feet model."""

from __future__ import annotations

import numpy as np

from athletic_analysis.core.pose.base import PoseBackend


def select_person(keypoints: np.ndarray, scores: np.ndarray,
                  last_center: np.ndarray | None, min_conf: float,
                  img_diag: float) -> int | None:
    """Pick the athlete among detected people, or None if nobody is credible.

    Detections below `min_conf` mean keypoint score are rejected outright —
    this is what keeps skeletons off background objects when nobody is in
    frame. Among credible people, prefer the one closest to where the athlete
    was last seen (temporal consistency), weighed against confidence.
    """
    mean_scores = np.asarray(scores).mean(axis=1)
    ok = np.where(mean_scores >= min_conf)[0]
    if len(ok) == 0:
        return None
    if last_center is None or img_diag <= 0:
        return int(ok[np.argmax(mean_scores[ok])])
    centers = np.asarray(keypoints)[ok].mean(axis=1)  # (n, 2)
    dist = np.linalg.norm(centers - last_center, axis=1) / img_diag
    return int(ok[np.argmax(mean_scores[ok] - 0.5 * dist)])


class RTMPoseBackend(PoseBackend):
    num_keypoints = 26

    def __init__(self, mode: str = "balanced", device: str = "cpu",
                 det_frequency: int = 5, min_person_conf: float = 0.35):
        # Import here so the rest of the app (tests, exports) works without rtmlib.
        from rtmlib import BodyWithFeet

        self._min_person_conf = min_person_conf
        self._last_center: np.ndarray | None = None
        # The person detector dominates CPU time; for a single athlete it's safe
        # to rerun it only every few frames (poses are tracked in between).
        try:
            self._model = BodyWithFeet(mode=mode, backend="onnxruntime",
                                       device=device, det_frequency=det_frequency)
        except TypeError:
            self._model = BodyWithFeet(mode=mode, backend="onnxruntime", device=device)

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        keypoints, scores = self._model(frame_bgr)
        if keypoints is None or len(keypoints) == 0:
            return self.empty()
        h, w = frame_bgr.shape[:2]
        best = select_person(keypoints, scores, self._last_center,
                             self._min_person_conf, float(np.hypot(h, w)))
        if best is None:
            return self.empty()
        kpts = np.zeros((self.num_keypoints, 3), dtype=np.float32)
        n = min(self.num_keypoints, keypoints.shape[1])
        kpts[:n, :2] = keypoints[best, :n]
        kpts[:n, 2] = scores[best, :n]
        self._last_center = np.asarray(keypoints[best], dtype=np.float64).mean(axis=0)
        return kpts
