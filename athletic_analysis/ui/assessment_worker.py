"""Background video-suitability assessment (runs on open, a few seconds)."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal


class AssessmentWorker(QThread):
    finished_ok = Signal(object)  # VideoAssessment
    failed = Signal(str)

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self._video_path = video_path

    def run(self) -> None:
        try:
            # Detector construction / first use may download+load ONNX; keep it
            # off the UI thread.
            from athletic_analysis.core.assessment import assess_video
            from athletic_analysis.core.pose.detector import PersonDetector

            detector = PersonDetector()
            self.finished_ok.emit(
                assess_video(self._video_path, detector=detector))
        except Exception:
            self.failed.emit(traceback.format_exc())
