"""Key-frame filmstrip: thumbnails of the moments that matter (touchdowns for
sprints; countermovement bottom / takeoff / landing for jumps) with the pose
drawn and a caption, for side-by-side posture comparison. Click to seek."""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QScrollArea,
                               QVBoxLayout, QWidget)

THUMB_HEIGHT = 130


class _Thumb(QFrame):
    clicked = Signal(int)

    def __init__(self, frame_idx: int, pixmap: QPixmap, caption: str, parent=None):
        super().__init__(parent)
        self._frame_idx = frame_idx
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        image = QLabel()
        image.setPixmap(pixmap)
        layout.addWidget(image)
        text = QLabel(caption)
        text.setStyleSheet("font-size: 10px;")
        text.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(text)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self._frame_idx)


class KeyframeStrip(QWidget):
    frame_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._inner = QWidget()
        self._row = QHBoxLayout(self._inner)
        self._row.setContentsMargins(2, 2, 2, 2)
        self._row.addStretch(1)
        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll)
        self._thumbs: list[_Thumb] = []
        self._empty = QLabel("Key frames appear here after analysis.")
        self._empty.setStyleSheet("color: gray;")
        self._row.insertWidget(0, self._empty)

    def set_keyframes(self, entries: list[tuple[int, str]],
                      render: Callable[[int], np.ndarray | None]) -> None:
        """entries: (frame_index, caption). `render` returns an annotated BGR
        frame for an index (or None if unavailable)."""
        for thumb in self._thumbs:
            self._row.removeWidget(thumb)
            thumb.deleteLater()
        self._thumbs = []
        self._empty.setVisible(not entries)
        for pos, (frame_idx, caption) in enumerate(entries):
            bgr = render(frame_idx)
            if bgr is None:
                continue
            h, w, _ = bgr.shape
            scale = THUMB_HEIGHT / h
            small = cv2.resize(bgr, (max(1, int(w * scale)), THUMB_HEIGHT))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            image = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                           3 * rgb.shape[1], QImage.Format.Format_RGB888).copy()
            thumb = _Thumb(frame_idx, QPixmap.fromImage(image), caption)
            thumb.clicked.connect(self.frame_requested.emit)
            self._row.insertWidget(pos, thumb)
            self._thumbs.append(thumb)
