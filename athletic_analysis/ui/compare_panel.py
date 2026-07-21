"""Compare tab: all of a sprint's flagged moments on ONE scrubbable video.

Instead of a gallery of disconnected per-fault loops, the tab shows a single
preview + scrub bar (`CompareScrubber`): every fault is a marker on the bar,
dragging seeks through the real footage (cropped to the athlete) and updates
a caption to the nearest flagged moment. Below it, a light list of fault
rows — each just its headline text and cue, plus a small schematic reference
clip *only* when no real step in the clip did that check well (nothing real
to point the scrubber at).

The Form tab reuses `ComparisonImages` (the two-clip side-by-side) unchanged
in shape, since it only ever shows one comparison at a time and isn't the
"separated instances" problem this rework targets.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QScrollArea,
                               QVBoxLayout, QWidget)

from athletic_analysis.core import reference_pose
from athletic_analysis.core.compare import StepComparison
from athletic_analysis.ui import theme
from athletic_analysis.ui.replay_clip import ReplayClip, to_pixmap
from athletic_analysis.ui.timeline import Timeline

THUMB_H = 150
PREVIEW_H = 230

# The Form tab hands ComparisonImages a list-returning renderer (a short
# clip); the Compare-tab scrubber hands its widgets a single-frame renderer.
RenderReplay = Callable[[int], list[np.ndarray]]
RenderFrame = Callable[[int], "np.ndarray | None"]


def _fmt_value(value: float | None, unit: str) -> str:
    if value is None or not np.isfinite(value):
        return "–"
    if unit == "deg":
        return f"{value:.0f}°"
    return f"{value:.2f} {unit}"


def _range_text(check) -> str:
    if check.unit == "deg":
        return f"{check.lo:.0f}–{check.hi:.0f}°"
    return f"{check.lo:.2f}–{check.hi:.2f} {check.unit}"


def _build_reference_clip(check, frames: list[np.ndarray]) -> ReplayClip:
    """The dashed schematic clip shown when no real step qualifies. Shared by
    the Compare-tab rows and the Form-tab ComparisonImages."""
    clip = ReplayClip(border_color=theme.ACCENT, dashed=True, height=THUMB_H,
                      caption=f"Target: {_range_text(check)}")
    clip.set_frames(frames)
    return clip


# --- Form-tab side-by-side (one comparison) ------------------------------

class ComparisonImages(QWidget):
    """Two clips side by side for a single fault: the athlete's flagged
    moment vs. their own best step, or vs. the animated schematic when no
    real step qualifies. Used inline by the Form tab."""

    frame_clicked = Signal(int)

    def __init__(self, comparison: StepComparison, render_replay: RenderReplay,
                parent=None):
        super().__init__(parent)
        f = comparison.finding
        images = QHBoxLayout(self)
        images.setContentsMargins(0, 0, 0, 0)

        images.addWidget(self._real_side(
            "You", f.frame, f.value_text, render_replay,
            theme.SEVERITY_COLORS.get(f.severity, theme.WARN)))
        trailing_stretch = True
        if comparison.best_frame is not None:
            images.addWidget(self._real_side(
                "Best", comparison.best_frame,
                _fmt_value(comparison.best_value, comparison.check.unit),
                render_replay, theme.GOOD))
        elif comparison.posable:
            target = (comparison.check.lo + comparison.check.hi) / 2
            frames = reference_pose.render_sequence(f.key, target, "°")
            images.addWidget(_build_reference_clip(comparison.check, frames))
        else:
            note = QLabel("No step in this clip pulled this one off, and it's a "
                          "timing/distance check — there's no single pose to "
                          "show as a reference.")
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {theme.hexs(theme.TEXT_MUTED)}; font-size: 10px;")
            images.addWidget(note, stretch=1)
            trailing_stretch = False
        if trailing_stretch:
            images.addStretch(1)

    def _real_side(self, label: str, frame: int, value_text: str,
                   render_replay: RenderReplay, border_color: theme.Rgb) -> ReplayClip:
        clip = ReplayClip(border_color=border_color, height=THUMB_H, clickable=True,
                          caption=f"{label}: {value_text}")
        clip.set_frames(render_replay(frame))
        clip.clicked.connect(lambda fr=frame: self.frame_clicked.emit(fr))
        return clip


# --- Compare-tab scrubber + rows -----------------------------------------

class CompareScrubber(QWidget):
    """One preview frame + a Timeline scrub bar carrying a marker per fault
    (and per real best-step). Dragging renders the frame and captions the
    nearest fault. One-directional: it seeks the main video but the main
    video's own playback does not push back here (avoids re-cropping a frame
    on every playhead tick during Play)."""

    frame_requested = Signal(int)

    def __init__(self, render_frame: RenderFrame, parent=None):
        super().__init__(parent)
        self._render_frame = render_frame
        self._comparisons: list[StepComparison] = []

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        self._preview = QLabel("Drag the bar to review each flagged moment.")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(PREVIEW_H * 3 // 4, PREVIEW_H)
        self._preview.setStyleSheet(
            f"border: 1px solid {theme.hexs(theme.SURFACE_RAISED)}; border-radius: 8px; "
            f"background: {theme.hexs(theme.SURFACE_RAISED)}; "
            f"color: {theme.hexs(theme.TEXT_MUTED)};")
        # Center the preview so its border hugs the (portrait) crop instead of
        # stretching a wide box around a narrow figure — the exact "video
        # extends beyond where it's useful" complaint.
        preview_row = QHBoxLayout()
        preview_row.addStretch(1)
        preview_row.addWidget(self._preview)
        preview_row.addStretch(1)
        box.addLayout(preview_row)

        self._timeline = Timeline()
        self._timeline.frame_requested.connect(self._on_scrub)
        box.addWidget(self._timeline)

        self._caption = QLabel("")
        self._caption.setWordWrap(True)
        self._caption.setStyleSheet("font-size: 11px;")
        box.addWidget(self._caption)

    def set_frame_count(self, count: int) -> None:
        self._timeline.set_frame_count(count)

    def set_comparisons(self, comparisons: list[StepComparison]) -> None:
        self._comparisons = comparisons
        markers: list[tuple[int, object, str]] = []
        for c in comparisons:
            f = c.finding
            markers.append((f.frame, theme.qcolor(
                theme.SEVERITY_COLORS.get(f.severity, theme.WARN)), f.metric))
            if c.best_frame is not None:
                markers.append((c.best_frame, theme.qcolor(theme.GOOD),
                                f"good {f.metric}"))
        self._timeline.set_markers(markers)
        if comparisons:
            self.show_frame(comparisons[0].finding.frame)

    def show_frame(self, frame: int) -> None:
        self._timeline.set_current(frame)
        img = self._render_frame(frame)
        if img is not None:
            pix = to_pixmap(img).scaledToHeight(
                PREVIEW_H, Qt.TransformationMode.SmoothTransformation)
            self._preview.setPixmap(pix)
            self._preview.setFixedSize(pix.size())  # border hugs the crop
        self._update_caption(frame)

    def _on_scrub(self, frame: int) -> None:
        self.show_frame(frame)
        self.frame_requested.emit(frame)

    def _update_caption(self, frame: int) -> None:
        if not self._comparisons:
            self._caption.setText("")
            return
        nearest = min(self._comparisons,
                      key=lambda c: abs(c.finding.frame - frame))
        f = nearest.finding
        color = theme.hexs(theme.SEVERITY_COLORS.get(f.severity, theme.WARN))
        self._caption.setText(
            f"<b style='color:{color}'>{f.metric} · {f.phase}</b> — "
            f"{f.value_text} vs optimal {f.target_text}. {f.cue}")


class _CompareRow(QFrame):
    """One fault's headline + cue, click-to-seek. Carries the schematic
    reference clip only when there's no real best step to scrub to."""

    frame_clicked = Signal(int)

    def __init__(self, comparison: StepComparison, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        f = comparison.finding
        self._frame = f.frame
        outer = QVBoxLayout(self)

        sev_color = theme.hexs(theme.SEVERITY_COLORS.get(f.severity, theme.WARN))
        title = QLabel(f"{f.metric} · {f.phase} — {f.value_text} "
                       f"(optimal {f.target_text})")
        title.setWordWrap(True)
        title.setStyleSheet(f"font-weight: 600; color: {sev_color}; border: none;")
        outer.addWidget(title)

        cue = QLabel(f.cue)
        cue.setWordWrap(True)
        cue.setStyleSheet(f"color: {theme.hexs(theme.TEXT_MUTED)}; font-size: 11px; border: none;")
        outer.addWidget(cue)

        # Only add a schematic clip when there's no real footage to scrub to.
        if comparison.best_frame is None and comparison.posable:
            target = (comparison.check.lo + comparison.check.hi) / 2
            frames = reference_pose.render_sequence(f.key, target, "°")
            row = QHBoxLayout()
            row.addWidget(_build_reference_clip(comparison.check, frames))
            row.addStretch(1)
            outer.addLayout(row)

    def mousePressEvent(self, event) -> None:
        self.frame_clicked.emit(self._frame)
        super().mousePressEvent(event)


class ComparePanel(QWidget):
    """Compare tab: a persistent scrubber over the clip's flagged moments,
    with a light fault list below. Sprint-only — jump clips don't have the
    repeated per-phase steps this comparison needs (main_window passes an
    explaining empty_message in that case)."""

    frame_requested = Signal(int)

    def __init__(self, render_frame: RenderFrame, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)

        self._empty = QLabel("Run pose analysis on a sprint clip to review your "
                             "flagged moments here.")
        self._empty.setWordWrap(True)
        self._empty.setStyleSheet(f"color: {theme.hexs(theme.TEXT_MUTED)};")
        outer.addWidget(self._empty)

        self._scrubber = CompareScrubber(render_frame)
        self._scrubber.frame_requested.connect(self.frame_requested)
        self._scrubber.hide()
        outer.addWidget(self._scrubber)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.hide()
        self._inner = QWidget()
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.addStretch(1)
        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll, stretch=1)
        self._rows: list[_CompareRow] = []

    def set_frame_count(self, count: int) -> None:
        self._scrubber.set_frame_count(count)

    def set_comparisons(self, comparisons: list[StepComparison],
                        empty_message: str = "") -> None:
        for row in self._rows:
            self._inner_layout.removeWidget(row)
            row.deleteLater()
        self._rows = []
        if not comparisons:
            self._empty.setText(empty_message or
                "No form faults to review — nice rep.")
            self._empty.show()
            self._scrubber.hide()
            self._scroll.hide()
            return
        self._empty.hide()
        self._scrubber.show()
        self._scroll.show()
        self._scrubber.set_comparisons(comparisons)
        for comparison in comparisons:
            row = _CompareRow(comparison)
            row.frame_clicked.connect(self._on_row_clicked)
            self._inner_layout.insertWidget(self._inner_layout.count() - 1, row)
            self._rows.append(row)

    def _on_row_clicked(self, frame: int) -> None:
        self._scrubber.show_frame(frame)
        self.frame_requested.emit(frame)
