"""Aspect-fit display of a composed BGR frame, with optional point picking
(used for calibration). Overlay drawing happens in OpenCV before frames get
here, so this widget stays a dumb display surface."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget


class VideoWidget(QWidget):
    point_picked = Signal(float, float)  # frame coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._frame_size: tuple[int, int] = (0, 0)  # (w, h)
        self._picking = False
        self._pick_points: list[tuple[float, float]] = []
        self.setMinimumSize(480, 270)

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        self._frame_size = (w, h)
        self._image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.update()

    # --- calibration picking ------------------------------------------------

    def start_picking(self) -> None:
        self._picking = True
        self._pick_points = []
        self.setCursor(Qt.CursorShape.CrossCursor)

    def stop_picking(self) -> None:
        self._picking = False
        self._pick_points = []
        self.unsetCursor()
        self.update()

    # --- geometry -----------------------------------------------------------

    def _display_rect(self) -> QRect:
        """Aspect-fit rectangle of the frame inside the widget."""
        if self._image is None or self._frame_size[0] == 0:
            return QRect()
        w, h = self._frame_size
        scale = min(self.width() / w, self.height() / h)
        dw, dh = int(w * scale), int(h * scale)
        return QRect((self.width() - dw) // 2, (self.height() - dh) // 2, dw, dh)

    def _widget_to_frame(self, pos: QPoint) -> tuple[float, float] | None:
        rect = self._display_rect()
        if rect.isEmpty() or not rect.contains(pos):
            return None
        fx = (pos.x() - rect.x()) / rect.width() * self._frame_size[0]
        fy = (pos.y() - rect.y()) / rect.height() * self._frame_size[1]
        return fx, fy

    def _frame_to_widget(self, fx: float, fy: float) -> QPoint:
        rect = self._display_rect()
        x = rect.x() + fx / self._frame_size[0] * rect.width()
        y = rect.y() + fy / self._frame_size[1] * rect.height()
        return QPoint(int(x), int(y))

    # --- events ---------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if not self._picking:
            return
        pt = self._widget_to_frame(event.position().toPoint())
        if pt is None:
            return
        self._pick_points.append(pt)
        self.update()
        self.point_picked.emit(*pt)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._image is not None:
            painter.drawImage(self._display_rect(), self._image)
        if self._picking and self._pick_points:
            pen = QPen(QColor(255, 80, 80), 2)
            painter.setPen(pen)
            pts = [self._frame_to_widget(x, y) for x, y in self._pick_points]
            for p in pts:
                painter.drawEllipse(p, 4, 4)
            if len(pts) == 2:
                painter.drawLine(pts[0], pts[1])
        painter.end()
