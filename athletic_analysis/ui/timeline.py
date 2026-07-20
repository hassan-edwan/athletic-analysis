"""Scrub bar with colored event markers (foot strikes, toe-offs, takeoff, landing)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from athletic_analysis.ui.plot_panel import PHASE_COLORS


class Timeline(QWidget):
    frame_requested = Signal(int)
    phase_zoom_requested = Signal(int, int)  # start_frame, end_frame

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_count = 0
        self._current = 0
        self._markers: list[tuple[int, QColor, str]] = []  # (frame, color, tooltip)
        self._phases: list[tuple[int, int, str]] = []
        self.setFixedHeight(36)
        self.setMouseTracking(True)
        self.setToolTip("Click/drag to scrub · double-click a phase to zoom the graph")

    def set_frame_count(self, count: int) -> None:
        self._frame_count = max(0, count)
        self.update()

    def set_current(self, frame: int) -> None:
        self._current = frame
        self.update()

    def set_markers(self, markers: list[tuple[int, QColor, str]]) -> None:
        self._markers = markers
        self.update()

    def set_phases(self, phases: list[tuple[int, int, str]]) -> None:
        self._phases = phases
        self.update()

    def _frame_at(self, x: int) -> int:
        if self._frame_count <= 1:
            return 0
        frac = min(1.0, max(0.0, (x - 4) / max(1, self.width() - 8)))
        return round(frac * (self._frame_count - 1))

    def _x_of(self, frame: int) -> int:
        if self._frame_count <= 1:
            return 4
        return 4 + round(frame / (self._frame_count - 1) * (self.width() - 8))

    def mousePressEvent(self, event) -> None:
        self.frame_requested.emit(self._frame_at(int(event.position().x())))

    def mouseDoubleClickEvent(self, event) -> None:
        frame = self._frame_at(int(event.position().x()))
        for f0, f1, _name in self._phases:
            if f0 <= frame <= f1:
                self.phase_zoom_requested.emit(f0, f1)
                return

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.frame_requested.emit(self._frame_at(int(event.position().x())))

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 34))
        bar = self.rect().adjusted(4, 12, -4, -12)
        p.fillRect(bar, QColor(60, 60, 68))
        for f0, f1, name in self._phases:
            rgb = PHASE_COLORS.get(name, (160, 160, 160))
            x0, x1 = self._x_of(f0), self._x_of(f1)
            p.fillRect(x0, 4, max(1, x1 - x0), self.height() - 8,
                       QColor(*rgb, 60))
        for frame, color, _tip in self._markers:
            x = self._x_of(frame)
            p.setPen(QPen(color, 2))
            p.drawLine(x, 4, x, self.height() - 4)
        # playhead
        x = self._x_of(self._current)
        p.setPen(QPen(QColor(240, 240, 240), 2))
        p.drawLine(x, 2, x, self.height() - 2)
        p.end()
