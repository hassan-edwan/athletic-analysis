"""A short, slowed-down looping clip — the replay mechanism behind every
form comparison (compare_panel.py, form_panel.py's inline row expansion).
Cycles a fixed list of already-rendered BGR frames at a constant slow
playback rate, independent of the source capture fps — that constant slow
rate is what reads as "slow motion" without needing real slow-mo footage.

Deliberately dumb: this widget doesn't know where its frames come from —
real footage (main_window._render_keyframe_range) and the synthetic
reference sequence (core/reference_pose.render_sequence) both just hand it
a list[np.ndarray] and it loops whichever it's given the same way.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from athletic_analysis.ui import theme

DEFAULT_FPS = 8


def to_pixmap(bgr: np.ndarray) -> QPixmap:
    """BGR ndarray -> QPixmap. Public because both this widget and the
    Compare-tab scrubber preview need it."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class ReplayClip(QWidget):
    """Looping frame-sequence display. `set_frames([...])` starts the loop;
    an empty list shows a neutral placeholder instead of a stalled frame.
    Optionally clickable — real-footage replays emit `clicked` so a caller
    can seek the main video to this clip's center frame; the synthetic
    reference clip typically leaves this unconnected since it has no real
    frame to seek to. An optional `caption` draws a small dark pill badge in
    the bottom-left corner, on top of the frame."""

    clicked = Signal()

    def __init__(self, border_color: theme.Rgb = theme.ACCENT, dashed: bool = False,
                height: int = 160, clickable: bool = False,
                caption: str | None = None, parent=None):
        super().__init__(parent)
        self._height = height
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._image = QLabel("…")
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setMinimumHeight(height)
        self._image.setStyleSheet(
            f"border: 2px {'dashed' if dashed else 'solid'} {theme.hexs(border_color)}; "
            f"border-radius: 6px; background: {theme.hexs(theme.SURFACE_RAISED)}; "
            f"color: {theme.hexs(theme.TEXT_MUTED)};")
        if clickable:
            self._image.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self._image)

        # A caption parented directly onto the image label paints on top of it
        # (no layout), with a fixed dark-translucent fill so it stays legible
        # over any frame content — theme.make_chip's color-keyed fill would
        # not. Repositioned in _reposition_caption() once the image has a size.
        self._caption: QLabel | None = None
        if caption:
            self._caption = QLabel(caption, self._image)
            self._caption.setStyleSheet(
                "background: rgba(10, 10, 12, 180); color: #e2e3e6; "
                "border-radius: 6px; padding: 1px 6px; font-size: 10px; "
                "font-weight: 600;")
            self._caption.adjustSize()

        self._clickable = clickable
        self._pixmaps: list[QPixmap] = []
        self._index = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    def mousePressEvent(self, event) -> None:
        if self._clickable:
            self.clicked.emit()
        super().mousePressEvent(event)

    def set_frames(self, frames: list[np.ndarray], fps: int = DEFAULT_FPS) -> None:
        self._timer.stop()
        self._pixmaps = [to_pixmap(f) for f in frames]
        self._index = 0
        if not self._pixmaps:
            self._image.setPixmap(QPixmap())
            self._image.setText("no frames available")
            self._image.setMinimumWidth(120)
            return
        self._show_current()
        if len(self._pixmaps) > 1:
            self._timer.start(max(1, round(1000 / fps)))

    def _show_current(self) -> None:
        pix = self._pixmaps[self._index].scaledToHeight(
            self._height, Qt.TransformationMode.SmoothTransformation)
        self._image.setPixmap(pix)
        # Lock the label to the pixmap's real size so QHBoxLayout can't
        # stretch the frame across empty space — the "box wider than the
        # picture" complaint.
        self._image.setFixedSize(pix.size())
        self._reposition_caption()

    def _reposition_caption(self) -> None:
        if self._caption is None:
            return
        self._caption.adjustSize()
        margin = 6
        self._caption.move(margin,
                           self._image.height() - self._caption.height() - margin)
        self._caption.raise_()

    def _advance(self) -> None:
        if not self._pixmaps:
            return
        self._index = (self._index + 1) % len(self._pixmaps)
        self._show_current()

    def stop(self) -> None:
        self._timer.stop()
