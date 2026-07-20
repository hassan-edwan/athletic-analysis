"""Background pose-estimation pass over the whole video, with optional
preprocessing (rotation via VideoSource, reframe/enhance/deinterlace)."""

from __future__ import annotations

import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal

from athletic_analysis.core.preprocess import Preprocessor
from athletic_analysis.core.video_source import VideoSource

_DET_STRIDE = 3  # detect the athlete every Nth frame; interpolate between


class AnalysisWorker(QThread):
    progress = Signal(int, int)  # work units done, total
    finished_ok = Signal(object)  # np.ndarray (T, 26, 3)
    failed = Signal(str)

    def __init__(self, video_path: str, model_tier: str = "Balanced",
                 preprocessor: Preprocessor | None = None, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._model_tier = model_tier
        self._pre = preprocessor or Preprocessor()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from athletic_analysis.core.pose.rtmpose_backend import RTMPoseBackend
            from athletic_analysis.core.settings import TIER_TO_MODE

            mode = TIER_TO_MODE.get(self._model_tier, "balanced")
            backend = RTMPoseBackend(mode=mode)

            source = VideoSource(self._video_path, rotation=self._pre.rotation)
            total = source.frame_count if source.frame_count > 0 else 0
            det_units = total if self._pre.needs_detection_pass() else 0
            total_units = (total + det_units) or 1

            # --- phase 1: athlete detection pass (only if reframing) ---
            if self._pre.needs_detection_pass():
                if not self._build_tracker(source, total, total_units):
                    return  # cancelled

            # --- phase 2: pose pass on preprocessed frames ---
            results: list[np.ndarray] = []
            for idx, frame in source.iter_frames():
                if self._cancelled:
                    return
                proc, to_original = self._pre.process(frame, idx)
                kpts = backend.estimate(proc)
                kpts[:, :2] = to_original(kpts[:, :2])
                results.append(kpts)
                done = det_units + idx + 1
                if idx % 5 == 0 or idx == total - 1:
                    self.progress.emit(done, total_units)
            source.close()
            if not results:
                self.failed.emit("No frames could be decoded from the video.")
                return
            self.finished_ok.emit(np.stack(results))
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _build_tracker(self, source: VideoSource, total: int,
                       total_units: int) -> bool:
        """Detect the athlete every _DET_STRIDE frames; attach a ReframeTracker
        to the preprocessor. Returns False if cancelled."""
        from athletic_analysis.core.pose.detector import (PersonDetector,
                                                          box_center,
                                                          select_tracked_box)
        from athletic_analysis.core.preprocess import ReframeTracker

        detector = PersonDetector()
        boxes: list[np.ndarray | None] = []
        last_center: np.ndarray | None = None
        for idx, frame in source.iter_frames():
            if self._cancelled:
                return False
            if idx % _DET_STRIDE == 0:
                box = select_tracked_box(detector.detect(frame), last_center)
                if box is not None:
                    last_center = box_center(box)
                boxes.append(box)
            else:
                boxes.append(None)
            if idx % 5 == 0:
                self.progress.emit(idx + 1, total_units)
        self._pre.tracker = ReframeTracker(
            boxes=boxes, frame_w=source.width, frame_h=source.height)
        return True
