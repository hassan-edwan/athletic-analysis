"""Compare tab: each form fault paired with a real example of the athlete
doing it well — or, when nothing in the clip pulls it off, a schematic
diagram at the target angle. Answers "what should this actually look like?"
visually, side by side, and as a short slow-motion replay rather than a
still frame — a snapshot doesn't show enough about motion to teach it."""

from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QScrollArea,
                               QVBoxLayout, QWidget)

from athletic_analysis.core import reference_pose
from athletic_analysis.core.compare import StepComparison
from athletic_analysis.ui import theme
from athletic_analysis.ui.replay_clip import ReplayClip

THUMB_H = 150

# render_replay(center_frame) -> a short list of BGR frames around that
# frame (main_window._render_keyframe_range with the fps-based half-window
# already baked in) — the real-footage half of every comparison.
RenderReplay = Callable[[int], list[np.ndarray]]


def _fmt_value(value: float | None, unit: str) -> str:
    if value is None or not np.isfinite(value):
        return "–"
    if unit == "deg":
        return f"{value:.0f}°"
    return f"{value:.2f} {unit}"


class ComparisonImages(QWidget):
    """The reusable half of a comparison: the two (or one) ReplayClip sides
    side by side, with captions. Used both by the Compare tab's gallery
    cards below and inline by FormPanel's per-row expansion — one
    implementation, two entry points, so "no critique without a visual"
    holds wherever a fault is shown, not just in the dedicated tab."""

    frame_clicked = Signal(int)

    def __init__(self, comparison: StepComparison, render_replay: RenderReplay,
                parent=None):
        super().__init__(parent)
        f = comparison.finding
        images = QHBoxLayout(self)
        images.setContentsMargins(0, 0, 0, 0)

        images.addWidget(self._real_side("Your form", f.frame, f.value_text,
                                         render_replay, theme.SEVERITY_COLORS.get(
                                             f.severity, theme.WARN)))
        if comparison.best_frame is not None:
            images.addWidget(self._real_side(
                "Your best step", comparison.best_frame,
                _fmt_value(comparison.best_value, comparison.check.unit),
                render_replay, theme.GOOD))
        elif comparison.posable:
            target = (comparison.check.lo + comparison.check.hi) / 2
            frames = reference_pose.render_sequence(f.key, target, "°")
            images.addWidget(self._reference_side(frames, comparison.check))
        else:
            note = QLabel("No step in this clip pulled this one off, and it's a "
                          "timing/distance check — there's no single pose to "
                          "show as a reference.")
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {theme.hexs(theme.TEXT_MUTED)}; font-size: 10px;")
            images.addWidget(note, stretch=1)

    def _real_side(self, caption: str, frame: int, value_text: str,
                   render_replay: RenderReplay, border_color: theme.Rgb) -> QWidget:
        wrap = QWidget()
        box = QVBoxLayout(wrap)
        box.setContentsMargins(0, 0, 0, 0)
        clip = ReplayClip(border_color=border_color, height=THUMB_H, clickable=True)
        clip.set_frames(render_replay(frame))
        clip.clicked.connect(lambda fr=frame: self.frame_clicked.emit(fr))
        box.addWidget(clip)
        cap = QLabel(f"{caption}\n{value_text}")
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap.setStyleSheet("font-size: 10px;")
        box.addWidget(cap)
        return wrap

    def _reference_side(self, frames: list[np.ndarray], check) -> QWidget:
        wrap = QWidget()
        box = QVBoxLayout(wrap)
        box.setContentsMargins(0, 0, 0, 0)
        clip = ReplayClip(border_color=theme.ACCENT, dashed=True, height=THUMB_H)
        clip.set_frames(frames)
        box.addWidget(clip)
        lo_hi = f"{check.lo:.0f}–{check.hi:.0f}°" if check.unit == "deg" \
            else f"{check.lo:.2f}–{check.hi:.2f} {check.unit}"
        cap = QLabel(f"Reference (schematic)\noptimal {lo_hi}")
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap.setStyleSheet("font-size: 10px;")
        box.addWidget(cap)
        return wrap


class _CompareCard(QFrame):
    frame_clicked = Signal(int)

    def __init__(self, comparison: StepComparison, render_replay: RenderReplay,
                parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        f = comparison.finding
        outer = QVBoxLayout(self)

        sev_color = theme.hexs(theme.SEVERITY_COLORS.get(f.severity, theme.WARN))
        title = QLabel(f"{f.metric} · {f.phase}")
        title.setStyleSheet(f"font-weight: 600; color: {sev_color}; border: none;")
        outer.addWidget(title)

        images = ComparisonImages(comparison, render_replay)
        images.frame_clicked.connect(self.frame_clicked)
        outer.addWidget(images)

        cue = QLabel(f.cue)
        cue.setWordWrap(True)
        cue.setStyleSheet(f"color: {theme.hexs(theme.TEXT_MUTED)}; font-size: 11px; border: none;")
        outer.addWidget(cue)


class ComparePanel(QWidget):
    """One card per fault: the athlete's flagged step next to their own best
    step for that same check, or a schematic reference when no step in the
    clip qualifies. Sprint-only — jump clips don't have the repeated,
    per-phase steps this comparison needs."""

    frame_requested = Signal(int)

    def __init__(self, render_replay: RenderReplay, parent=None):
        super().__init__(parent)
        self._render_replay = render_replay
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
            card = _CompareCard(comparison, self._render_replay)
            card.frame_clicked.connect(self.frame_requested.emit)
            self._inner_layout.insertWidget(self._inner_layout.count() - 1, card)
            self._cards.append(card)
