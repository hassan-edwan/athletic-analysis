"""Merged Form + Compare review: one tab that grades the rep, plays the
footage with the *optimal* form drawn behind the runner, and quantifies how
far off they are — surfacing a frame as a poor-form instance (with root-cause
diagnosis) when the deviation is significant.

Select a check in the summary table; the scrubbable preview then overlays the
corrected 'ghost' segment for that metric (a translucent version of the
athlete's own limb rotated to the nearest in-range angle) and, as you drag,
shows the live degrees-off and flags the frames where it's a major fault.
Timing/rhythm checks (ground contact, cadence, overstride) have no single pose
to correct, so they show their measured-vs-optimal readout and diagnosis
without a ghost overlay.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QTableWidget,
                               QTableWidgetItem, QTextBrowser, QVBoxLayout,
                               QWidget)

from athletic_analysis.core import form_overlay as fo
from athletic_analysis.core.coaching import FormFinding, summarize
from athletic_analysis.core.diagnostics import Diagnosis, diagnose
from athletic_analysis.ui import theme
from athletic_analysis.ui.replay_clip import to_pixmap
from athletic_analysis.ui.timeline import Timeline

PREVIEW_H = 260

# render_review_frame(frame, spec, severity, off) -> BGR frame (pose + overlay,
# cropped to the athlete) or None.
RenderReview = Callable[[int, "fo.OverlaySpec | None", str, float], "np.ndarray | None"]

_SEVERITY_BG = {k: theme.qcolor(v, 55) for k, v in theme.SEVERITY_COLORS.items()}
_SEVERITY_LABEL = {"good": "OK", "minor": "Minor", "major": "Fix"}


class FormReviewPanel(QWidget):
    frame_requested = Signal(int)

    def __init__(self, render_review_frame: RenderReview, parent=None):
        super().__init__(parent)
        self._render = render_review_frame
        self._findings: list[FormFinding] = []
        self._row_findings: list[FormFinding] = []
        self._kpts: np.ndarray | None = None
        self._angles: dict = {}
        self._spans: list[tuple[int, int, str]] = []
        self._level = "trained"
        self._direction = 1.0
        self._active_key = ""      # posable metric currently overlaid ("" = none)
        self._active_finding: FormFinding | None = None

        layout = QVBoxLayout(self)
        self._summary = QLabel("Run pose analysis to review your form.")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setMaximumHeight(190)
        self._table.currentCellChanged.connect(self._on_row_changed)
        layout.addWidget(self._table)

        # Centered, content-hugging preview (see replay_clip for why fixed-size).
        self._preview = QLabel("Select a check above to review it here.")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(PREVIEW_H * 3 // 4, PREVIEW_H)
        self._preview.setStyleSheet(
            f"border: 1px solid {theme.hexs(theme.SURFACE_RAISED)}; border-radius: 8px; "
            f"background: {theme.hexs(theme.SURFACE_RAISED)}; "
            f"color: {theme.hexs(theme.TEXT_MUTED)};")
        prow = QHBoxLayout()
        prow.addStretch(1)
        prow.addWidget(self._preview)
        prow.addStretch(1)
        layout.addLayout(prow)

        self._readout = QLabel("")
        self._readout.setWordWrap(True)
        self._readout.setTextFormat(Qt.TextFormat.RichText)
        self._readout.setStyleSheet("font-size: 12px;")
        layout.addWidget(self._readout)

        self._timeline = Timeline()
        self._timeline.frame_requested.connect(self._on_scrub)
        layout.addWidget(self._timeline)

        self._legend = QLabel(
            f"<span style='color:{theme.hexs(theme.GOOD)}'>Green ghost</span> = "
            "optimal form · the colored gap is how far off you are. Markers flag "
            "frames that are a major fault.")
        self._legend.setWordWrap(True)
        self._legend.setStyleSheet(f"color: {theme.hexs(theme.TEXT_MUTED)}; font-size: 10px;")
        layout.addWidget(self._legend)

        self._detail = QTextBrowser()
        self._detail.setOpenExternalLinks(False)
        layout.addWidget(self._detail, stretch=1)

    # --- population ------------------------------------------------------

    def clear(self, message: str) -> None:
        self._findings = []
        self._row_findings = []
        self._kpts = None
        self._active_key = ""
        self._active_finding = None
        self._summary.setText(message)
        self._table.clear()
        self._table.setRowCount(0)
        self._timeline.set_markers([])
        self._readout.setText("")
        self._detail.clear()
        self._preview.setPixmap(QPixmap())
        self._preview.setText("Select a check above to review it here.")

    def set_analysis(self, findings: list[FormFinding], kpts: np.ndarray,
                     angles: dict, spans: list[tuple[int, int, str]],
                     level: str, direction: float, frame_count: int) -> None:
        self._findings = findings
        self._kpts = kpts
        self._angles = angles
        self._spans = spans
        self._level = level
        self._direction = direction
        self._timeline.set_frame_count(frame_count)
        self._timeline.set_phases(spans)
        self._summary.setText(summarize(findings))
        self._populate_table(findings)
        # Auto-select the most severe fault (posable preferred) so the tab
        # opens on something meaningful.
        faults = [f for f in findings if f.severity != "good"]
        posable_faults = [f for f in faults if f.key in fo.POSABLE_KEYS]
        target = (posable_faults or faults or findings)
        if target:
            want = target[0]
            for r, f in enumerate(self._row_findings):
                if f is want:
                    self._table.setCurrentCell(r, 0)
                    break

    def _populate_table(self, findings: list[FormFinding]) -> None:
        headers = ["", "Metric", "Phase", "Measured", "Optimal"]
        self._table.clear()
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        # Faults first, then in-range checks.
        ordered = sorted(findings, key=lambda f: (f.severity == "good", f.frame))
        self._row_findings = ordered
        self._table.setRowCount(len(ordered))
        for r, f in enumerate(ordered):
            cells = [_SEVERITY_LABEL.get(f.severity, ""), f.metric, f.phase,
                     f.value_text, f.target_text]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setBackground(_SEVERITY_BG.get(f.severity, QColor(0, 0, 0, 0)))
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()

    # --- interaction -----------------------------------------------------

    def _on_row_changed(self, row: int, *_args) -> None:
        if not (0 <= row < len(self._row_findings)):
            return
        f = self._row_findings[row]
        self._active_finding = f
        self._active_key = f.key if f.key in fo.POSABLE_KEYS else ""
        self._refresh_markers()
        self._show_diagnosis(f)
        self.show_frame(f.frame)
        self.frame_requested.emit(f.frame)

    def _refresh_markers(self) -> None:
        if not self._active_key or self._kpts is None:
            # No live overlay: mark just the selected finding's frame.
            frame = self._active_finding.frame if self._active_finding else 0
            self._timeline.set_markers([(frame, theme.qcolor(theme.WARN), "")])
            return
        frames = fo.major_frames(self._active_key, self._kpts, self._angles,
                                 self._spans, self._level, self._direction)
        # A metric that's major over a long contiguous stretch would draw a
        # marker per frame — a solid block. Thin to ~60 evenly-spaced so the
        # distribution reads without overwhelming the bar (no-op when sparse).
        if len(frames) > 60:
            idx = np.linspace(0, len(frames) - 1, 60).round().astype(int)
            frames = [frames[i] for i in sorted(set(idx))]
        color = theme.qcolor(theme.BAD)
        self._timeline.set_markers([(fr, color, "major") for fr in frames])

    def _on_scrub(self, frame: int) -> None:
        self.show_frame(frame)
        self.frame_requested.emit(frame)

    def show_frame(self, frame: int) -> None:
        self._timeline.set_current(frame)
        spec = None
        severity = "good"
        off = 0.0
        if self._active_key and self._kpts is not None:
            ev = fo.live_eval(self._active_key, frame, self._kpts, self._angles,
                              self._spans, self._level, self._direction)
            if ev is not None:
                spec, severity, off = ev.spec, ev.severity, ev.off
                self._set_readout_live(ev)
            else:
                self._set_readout_static()
        else:
            self._set_readout_static()
        img = self._render(frame, spec, severity, off)
        if img is not None:
            pix = to_pixmap(img).scaledToHeight(
                PREVIEW_H, Qt.TransformationMode.SmoothTransformation)
            self._preview.setPixmap(pix)
            self._preview.setFixedSize(pix.size())

    def _set_readout_live(self, ev: "fo.LiveEval") -> None:
        color = theme.hexs(theme.SEVERITY_COLORS.get(ev.severity, theme.WARN))
        band = f"{ev.lo:.0f}–{ev.hi:.0f}°"
        if ev.off < 1.0:
            gap = "<span style='color:%s'>on target</span>" % theme.hexs(theme.GOOD)
        else:
            gap = (f"<b style='color:{color}'>{ev.off:.0f}° "
                   f"{'over' if ev.over else 'under'}</b>")
        tag = ""
        if ev.severity == "major":
            tag = (f" <b style='color:{theme.hexs(theme.BAD)}'>· POOR FORM</b>")
        metric = self._active_finding.metric if self._active_finding else ev.key
        self._readout.setText(
            f"{metric} · {ev.phase} — <b>{ev.value:.0f}°</b> vs optimal "
            f"{band} · {gap}{tag}")

    def _set_readout_static(self) -> None:
        f = self._active_finding
        if f is None:
            self._readout.setText("")
            return
        color = theme.hexs(theme.SEVERITY_COLORS.get(f.severity, theme.WARN))
        note = "" if f.key in fo.POSABLE_KEYS else \
            " <span style='color:%s'>(no pose overlay for timing/rhythm checks)</span>" \
            % theme.hexs(theme.TEXT_MUTED)
        self._readout.setText(
            f"<b style='color:{color}'>{f.metric} · {f.phase}</b> — "
            f"{f.value_text} vs optimal {f.target_text}{note}")

    def _show_diagnosis(self, finding: FormFinding) -> None:
        diag = diagnose(finding)
        if diag is None:
            if finding.severity == "good":
                self._detail.setHtml(
                    f"<i style='color:{theme.hexs(theme.GOOD)}'>In the optimal "
                    "range — nothing to fix here.</i>")
            else:
                self._detail.setHtml(
                    "<i>No root-cause entry for this deviation — usually a "
                    "tracking or capture-FPS issue rather than a form fault.</i>")
            return
        self._detail.setHtml(self._diagnosis_html(finding, diag))

    @staticmethod
    def _diagnosis_html(finding: FormFinding, diag: Diagnosis) -> str:
        def bullets(items) -> str:
            lis = "".join(f"<li>{i}</li>" for i in items)
            return f"<ul style='margin: 2px 0 6px 16px;'>{lis}</ul>"

        color = theme.hexs(theme.SEVERITY_COLORS.get(finding.severity, theme.WARN))
        muted = theme.hexs(theme.TEXT_MUTED)
        parts = [f"<b style='color:{color};'>{diag.title}</b>"
                 f" <span style='color:{muted};'>({finding.phase} · measured "
                 f"{finding.value_text} vs optimal {finding.target_text})</span>",
                 "<b>Why this happens</b>", bullets(diag.technical_causes),
                 "<b>Likely physical limiters</b>", bullets(diag.muscle_factors),
                 "<b>Corrective drills</b>", bullets(diag.drills)]
        if diag.phase_note:
            parts.append(f"<i style='color:{muted};'>{diag.phase_note}</i>")
        if diag.source:
            parts.append(f"<span style='color:{muted}; font-size:10px;'>Source: "
                         f"{diag.source}</span>")
        return "<br>".join(parts)
