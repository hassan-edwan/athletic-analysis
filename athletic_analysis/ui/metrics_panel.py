"""Metrics display: summary stats + per-step (sprint) or per-jump details."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QLabel, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from athletic_analysis.core.metrics.jump import JumpMetrics
from athletic_analysis.core.metrics.sprint import SprintMetrics


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
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.cellClicked.connect(self._on_cell_clicked)
        layout.addWidget(self._table)
        self._row_frames: list[int] = []

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._row_frames):
            self.frame_requested.emit(self._row_frames[row])

    def show_sprint(self, m: SprintMetrics | None) -> None:
        if m is None or not m.steps:
            self._summary.setText("No steps detected yet.")
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
        self._table.setRowCount(len(m.steps))
        self._row_frames = [s.strike_frame for s in m.steps]
        for r, s in enumerate(m.steps):
            cells = [
                s.side, str(s.strike_frame),
                _fmt(s.contact_time_s * 1000, 0), _fmt(s.flight_time_s * 1000, 0),
                _fmt(s.step_length), _fmt(s.step_speed),
                _fmt(s.knee_angle_at_strike, 0),
                _fmt(s.swing_thigh_angle, 0), _fmt(s.trunk_lean_at_strike, 0),
            ]
            for c, text in enumerate(cells):
                self._table.setItem(r, c, QTableWidgetItem(text))
        self._table.resizeColumnsToContents()

    def show_jump(self, m: JumpMetrics | None) -> None:
        self._table.clear()
        self._row_frames = []
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
