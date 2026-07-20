"""Compare tab: each form fault paired with a real example of the athlete
doing it well — or, when nothing in the clip pulls it off, a schematic
diagram at the target angle. Answers "what should this actually look like?"
visually, side by side, instead of leaving the athlete to interpret a number
against a range on their own."""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QScrollArea,
                               QVBoxLayout, QWidget)

from athletic_analysis.core import reference_pose
from athletic_analysis.core.compare import StepComparison
from athletic_analysis.ui import theme

THUMB_H = 150


def _to_pixmap(bgr: np.ndarray) -> QPixmap:
    h, w = bgr.shape[:2]
    scale = THUMB_H / h
    small = cv2.resize(bgr, (max(1, int(w * scale)), THUMB_H))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1],
                   QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


def _fmt_value(value: float | None, unit: str) -> str:
    if value is None or not np.isfinite(value):
        return "–"
    if unit == "deg":
        return f"{value:.0f}°"
    return f"{value:.2f} {unit}"


class _ClickableImage(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()
        super().mousePressEvent(event)


class _CompareCard(QFrame):
    frame_clicked = Signal(int)

    def __init__(self, comparison: StepComparison,
                render_keyframe: Callable[[int], np.ndarray | None], parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        f = comparison.finding
        outer = QVBoxLayout(self)

        sev_color = theme.hexs(theme.SEVERITY_COLORS.get(f.severity, theme.WARN))
        title = QLabel(f"{f.metric} · {f.phase}")
        title.setStyleSheet(f"font-weight: 600; color: {sev_color}; border: none;")
        outer.addWidget(title)

        images = QHBoxLayout()
        images.addWidget(self._real_side("Your form", f.frame, f.value_text,
                                         render_keyframe, sev_color))
        if comparison.best_frame is not None:
            images.addWidget(self._real_side(
                "Your best step", comparison.best_frame,
                _fmt_value(comparison.best_value, comparison.check.unit),
                render_keyframe, theme.hexs(theme.GOOD)))
        elif comparison.posable:
            target = (comparison.check.lo + comparison.check.hi) / 2
            img = reference_pose.render(f.key, target, "°")
            images.addWidget(self._reference_side(img, comparison.check))
        else:
            note = QLabel("No step in this clip pulled this one off, and it's a "
                          "timing/distance check — there's no single pose to "
                          "show as a reference.")
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {theme.hexs(theme.TEXT_MUTED)}; font-size: 10px; border: none;")
            images.addWidget(note, stretch=1)
        outer.addLayout(images)

        cue = QLabel(f.cue)
        cue.setWordWrap(True)
        cue.setStyleSheet(f"color: {theme.hexs(theme.TEXT_MUTED)}; font-size: 11px; border: none;")
        outer.addWidget(cue)

    def _real_side(self, caption: str, frame: int, value_text: str,
                   render_keyframe: Callable[[int], np.ndarray | None],
                   border_color: str) -> QWidget:
        wrap = QWidget()
        box = QVBoxLayout(wrap)
        box.setContentsMargins(0, 0, 0, 0)
        bgr = render_keyframe(frame)
        img = _ClickableImage()
        img.setCursor(Qt.CursorShape.PointingHandCursor)
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if bgr is not None:
            img.setPixmap(_to_pixmap(bgr))
        img.setStyleSheet(f"border: 2px solid {border_color}; border-radius: 6px;")
        img.clicked.connect(lambda fr=frame: self.frame_clicked.emit(fr))
        box.addWidget(img)
        cap = QLabel(f"{caption}\n{value_text}")
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap.setStyleSheet("font-size: 10px; border: none;")
        box.addWidget(cap)
        return wrap

    def _reference_side(self, bgr: np.ndarray, check) -> QWidget:
        wrap = QWidget()
        box = QVBoxLayout(wrap)
        box.setContentsMargins(0, 0, 0, 0)
        img = QLabel()
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img.setPixmap(_to_pixmap(bgr))
        img.setStyleSheet(f"border: 2px dashed {theme.hexs(theme.ACCENT)}; border-radius: 6px;")
        box.addWidget(img)
        lo_hi = f"{check.lo:.0f}–{check.hi:.0f}°" if check.unit == "deg" \
            else f"{check.lo:.2f}–{check.hi:.2f} {check.unit}"
        cap = QLabel(f"Reference (schematic)\noptimal {lo_hi}")
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap.setStyleSheet("font-size: 10px; border: none;")
        box.addWidget(cap)
        return wrap


class ComparePanel(QWidget):
    """One card per fault: the athlete's flagged step next to their own best
    step for that same check, or a schematic reference when no step in the
    clip qualifies. Sprint-only — jump clips don't have the repeated,
    per-phase steps this comparison needs."""

    frame_requested = Signal(int)

    def __init__(self, render_keyframe: Callable[[int], np.ndarray | None], parent=None):
        super().__init__(parent)
        self._render_keyframe = render_keyframe
        outer = QVBoxLayout(self)

        self._empty = QLabel("Run pose analysis on a sprint clip to compare your "
                             "form against a good example.")
        self._empty.setWordWrap(True)
        self._empty.setStyleSheet(f"color: {theme.hexs(theme.TEXT_MUTED)};")
        outer.addWidget(self._empty)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.hide()
        self._inner = QWidget()
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.addStretch(1)
        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll)
        self._cards: list[_CompareCard] = []

    def set_comparisons(self, comparisons: list[StepComparison], empty_message: str = "") -> None:
        for card in self._cards:
            self._inner_layout.removeWidget(card)
            card.deleteLater()
        self._cards = []
        if not comparisons:
            self._empty.setText(empty_message or
                "No form faults to compare against — nice rep.")
            self._empty.show()
            self._scroll.hide()
            return
        self._empty.hide()
        self._scroll.show()
        for comparison in comparisons:
            card = _CompareCard(comparison, self._render_keyframe)
            card.frame_clicked.connect(self.frame_requested.emit)
            self._inner_layout.insertWidget(self._inner_layout.count() - 1, card)
            self._cards.append(card)
