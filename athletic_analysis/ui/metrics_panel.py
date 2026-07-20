"""Metrics display: summary stats + per-step (sprint) or per-jump details.

The per-step table used to be a bare wall of numbers — accurate, but you had
to already know the optimal ranges to tell a 145 ms contact from a 95 ms one.
Cells for the checked columns (contact time, knee/thigh/trunk angles) are now
tinted with the same good/minor/major coloring the Form tab uses, evaluated
per step against whichever phase that step actually fell in — not just the
phase-median FormPanel grades on, so a step can visibly disagree with the
clip's overall verdict.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QLabel, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from athletic_analysis.core.coaching import (PhaseBucket, _evaluate,
                                             metric_help, sprint_checks)
from athletic_analysis.core.metrics.jump import JumpMetrics
from athletic_analysis.core.metrics.sprint import SprintMetrics, StepRecord
from athletic_analysis.ui import theme

# Column index -> (coaching check key, raw value in the check's own units —
# seconds for contact, degrees for the angle checks).
_CHECKED_COLUMNS: dict[int, tuple[str, Callable[[StepRecord], float]]] = {
    2: ("contact_ms", lambda s: s.contact_time_s),
    6: ("knee_strike", lambda s: s.knee_angle_at_strike),
    7: ("thigh", lambda s: s.swing_thigh_angle),
    8: ("trunk", lambda s: s.trunk_lean_at_strike),
}

# Column index -> metric_help() key, for header tooltips. Columns without an
# entry (Side, Strike, Step len, Speed) aren't coaching-check metrics, so
# they get no tooltip rather than a made-up one.
_HEADER_HELP_KEY = {2: "contact_ms", 6: "knee_strike", 7: "thigh", 8: "trunk"}

# Nine flat columns read as noise; naming the three groups they fall into
# (identity / timing & kinematics / posture) does the same job "group the
# columns" was meant to. A per-column header background tint was tried
# first and dropped: QTableWidgetItem.setBackground() on a header item is
# unrenderable once theme.py's app-wide QSS gives QHeaderView::section a
# solid background — confirmed with an isolated repro (a bare QTableWidget,
# one header item painted pure red, still invisible), not just a styling
# nuance to tune further.
_GROUPS_CAPTION = (
    "Columns: Side, Strike (identity) · Contact ms, Flight ms, Step len, "
    "Speed (timing & kinematics) · Knee@strike, Swing thigh, Trunk (posture "
    "checks, hover for what each measures).")


def _fmt(value: float, decimals: int = 2, suffix: str = "") -> str:
    if value is None or not np.isfinite(value):
        return "–"
    return f"{value:.{decimals}f}{suffix}"


class MetricsPanel(QWidget):
    frame_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._summary = QLabel("Run pose analysis to see metrics.")
        self._summary.setWordWrap(True)
        self._summary.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._summary)
        self._legend = QLabel(
            f"Tinted cells: <span style='color:{theme.hexs(theme.GOOD)}'>in range</span> · "
            f"<span style='color:{theme.hexs(theme.WARN)}'>minor</span> · "
            f"<span style='color:{theme.hexs(theme.BAD)}'>major</span> deviation for "
            "that step's own phase — the number in parentheses is the "
            "optimal range. Hover a column header for what it measures.")
        self._legend.setTextFormat(Qt.TextFormat.RichText)
        self._legend.setWordWrap(True)
        self._legend.setStyleSheet("font-size: 10px;")
        layout.addWidget(self._legend)
        self._groups_caption = QLabel(_GROUPS_CAPTION)
        self._groups_caption.setWordWrap(True)
        self._groups_caption.setStyleSheet(
            f"font-size: 10px; color: {theme.hexs(theme.TEXT_MUTED)};")
        layout.addWidget(self._groups_caption)
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.cellClicked.connect(self._on_cell_clicked)
        layout.addWidget(self._table)
        self._row_frames: list[int] = []

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._row_frames):
            self.frame_requested.emit(self._row_frames[row])

    def _frame_phase(self, buckets: dict[str, PhaseBucket]) -> dict[int, str]:
        out: dict[int, str] = {}
        for phase, bucket in buckets.items():
            for frame in bucket.strike_frames:
                out[frame] = phase
        return out

    def show_sprint(self, m: SprintMetrics | None,
                    buckets: dict[str, PhaseBucket] | None = None,
                    level: str = "trained") -> None:
        if m is None or not m.steps:
            self._summary.setText("No steps detected yet.")
            self._legend.hide()
            self._table.clear()
            self._table.setRowCount(0)
            self._row_frames = []
            return
        unit = m.length_unit
        self._summary.setText(
            f"<b>Sprint</b> — cadence <b>{_fmt(m.cadence_spm, 0)}</b> steps/min · "
            f"avg speed <b>{_fmt(m.mean_speed)} {unit}/s</b> · "
            f"top speed <b>{_fmt(m.max_speed)} {unit}/s</b> · "
            f"contact <b>{_fmt(m.mean_contact_s * 1000, 0)} ms</b> · "
            f"flight <b>{_fmt(m.mean_flight_s * 1000, 0)} ms</b> · "
            f"step length <b>{_fmt(m.mean_step_length)} {unit}</b> · "
            f"trunk lean <b>{_fmt(m.mean_trunk_lean_deg, 0)}°</b>"
            + ("" if unit == "m" else "<br><i>Lengths in body-heights — calibrate for meters.</i>"))
        headers = ["Side", "Strike", "Contact ms", "Flight ms",
                   f"Step len {unit}", f"Speed {unit}/s",
                   "Knee@strike °", "Swing thigh °", "Trunk °"]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._style_headers()
        self._table.setRowCount(len(m.steps))
        self._row_frames = [s.strike_frame for s in m.steps]

        frame_phase = self._frame_phase(buckets) if buckets else {}
        checks_by_phase = sprint_checks(level) if buckets else {}
        self._legend.show()

        for r, step in enumerate(m.steps):
            cells = [
                step.side, str(step.strike_frame),
                _fmt(step.contact_time_s * 1000, 0), _fmt(step.flight_time_s * 1000, 0),
                _fmt(step.step_length), _fmt(step.step_speed),
                _fmt(step.knee_angle_at_strike, 0),
                _fmt(step.swing_thigh_angle, 0), _fmt(step.trunk_lean_at_strike, 0),
            ]
            phase = frame_phase.get(step.strike_frame)
            checks = dict(checks_by_phase.get(phase, [])) if phase else {}
            for c, text in enumerate(cells):
                key_getter = _CHECKED_COLUMNS.get(c)
                if key_getter is not None and phase is not None:
                    key, getter = key_getter
                    check = checks.get(key)
                    if check is not None:
                        finding = _evaluate(check, getter(step), phase, step.strike_frame)
                        if finding is not None:
                            item = QTableWidgetItem(f"{text} ({finding.target_text})")
                            item.setBackground(theme.qcolor(
                                theme.SEVERITY_COLORS[finding.severity], 55))
                            item.setToolTip(f"{finding.metric} ({phase}): optimal "
                                            f"{finding.target_text}")
                            self._table.setItem(r, c, item)
                            continue
                self._table.setItem(r, c, QTableWidgetItem(text))
        self._table.resizeColumnsToContents()

    def _style_headers(self) -> None:
        """Attach a plain-language tooltip to the columns that are actually
        coaching checks (see _GROUPS_CAPTION's docstring for why this isn't
        also a per-column background tint)."""
        for c in range(self._table.columnCount()):
            item = self._table.horizontalHeaderItem(c)
            if item is None:
                continue
            help_key = _HEADER_HELP_KEY.get(c)
            if help_key:
                item.setToolTip(metric_help(help_key))

    def show_jump(self, m: JumpMetrics | None) -> None:
        self._table.clear()
        self._row_frames = []
        self._legend.hide()
        if m is None or m.takeoff_frame < 0:
            self._summary.setText("No jump detected yet.")
            self._table.setRowCount(0)
            return
        unit = m.length_unit
        self._summary.setText(
            f"<b>Jump</b> — height (flight time) <b>{_fmt(m.jump_height_flight_m)} m</b> · "
            f"flight <b>{_fmt(m.flight_time_s * 1000, 0)} ms</b> · "
            f"takeoff velocity <b>{_fmt(m.takeoff_velocity_m_s)} m/s</b>")
        rows = [
            ("Takeoff frame", str(m.takeoff_frame), m.takeoff_frame),
            ("Landing frame", str(m.landing_frame), m.landing_frame),
            ("Flight time", _fmt(m.flight_time_s * 1000, 0, " ms"), m.takeoff_frame),
            ("Jump height (g·t²/8)", _fmt(m.jump_height_flight_m, 2, " m"), m.takeoff_frame),
            ("Takeoff velocity (g·t/2)", _fmt(m.takeoff_velocity_m_s, 2, " m/s"), m.takeoff_frame),
            (f"Hip rise ({unit})", _fmt(m.hip_rise), m.takeoff_frame),
            (f"Countermovement depth ({unit})", _fmt(m.countermovement_depth), m.takeoff_frame),
            ("Knee angle at takeoff", _fmt(m.knee_angle_at_takeoff, 0, "°"), m.takeoff_frame),
            ("Hip angle at takeoff", _fmt(m.hip_angle_at_takeoff, 0, "°"), m.takeoff_frame),
            ("Trunk lean at takeoff", _fmt(m.trunk_lean_at_takeoff, 0, "°"), m.takeoff_frame),
            ("Peak knee flexion on landing", _fmt(m.peak_knee_flexion_landing, 0, "°"), m.landing_frame),
            ("Knee/ankle separation ratio (frontal)", _fmt(m.knee_ankle_sep_ratio_landing), m.landing_frame),
        ]
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._table.setRowCount(len(rows))
        for r, (name, value, frame) in enumerate(rows):
            self._table.setItem(r, 0, QTableWidgetItem(name))
            self._table.setItem(r, 1, QTableWidgetItem(value))
            self._row_frames.append(int(frame))
        self._table.resizeColumnsToContents()
