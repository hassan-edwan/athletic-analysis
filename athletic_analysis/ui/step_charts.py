"""Per-step bar charts: contact time, step length, step speed. Bars are
bordered by leg color (left/right) so asymmetry is visible at a glance;
contact time — the one of the three with an actual coaching check — is also
fill-tinted good/minor/major, same language as the Form and Metrics tabs,
with a shaded target-range zone per phase. Step length and speed have no
graded "optimal" in this app (they're informational, not checked against a
range), so they stay leg-colored only — no fabricated target band.
Clicking a bar seeks the video to that foot strike."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QCursor, QPen
from PySide6.QtWidgets import (QGraphicsRectItem, QHBoxLayout, QLabel,
                               QToolTip, QVBoxLayout, QWidget)

from athletic_analysis.core.coaching import (PhaseBucket, _evaluate,
                                             metric_help, sprint_checks)
from athletic_analysis.core.metrics.sprint import StepRecord
from athletic_analysis.ui import theme

_LEFT = theme.LEG_LEFT
_RIGHT = theme.LEG_RIGHT

_CAPTIONS = {
    "contact ms": metric_help("contact_ms"),
    "step length": "Stride distance for that step — longer isn't "
                   "automatically better, it should match your cadence "
                   "and speed.",
    "speed": "This step's average speed — watch for a dip that flags a "
             "weak step, not an absolute target.",
}


def _phase_spans_by_step_index(steps: list[StepRecord],
                               frame_phase: dict[int, str]
                               ) -> list[tuple[int, int, str]]:
    """Contiguous (start_idx, end_idx, phase) runs over `steps`' order —
    same idea as coaching.segment_phases, but keyed by step index instead
    of frame, and with no gap-merging since every index has a step."""
    spans: list[list] = []
    for i, step in enumerate(steps):
        phase = frame_phase.get(step.strike_frame)
        if phase is None:
            continue
        if spans and spans[-1][2] == phase and spans[-1][1] == i - 1:
            spans[-1][1] = i
        else:
            spans.append([i, i, phase])
    return [(a, b, p) for a, b, p in spans]


class _StepPlot(QWidget):
    """One captioned bar chart: a QLabel caption above a pyqtgraph PlotWidget."""

    def __init__(self, title: str, show_x_axis: bool, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.caption = QLabel("")
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet(f"color: {theme.hexs(theme.TEXT_MUTED)}; font-size: 10px;")
        layout.addWidget(self.caption)
        self.plot_widget = pg.PlotWidget()
        self.plot = self.plot_widget.getPlotItem()
        self.plot.setLabel("left", title)
        self.plot.getAxis("bottom").setStyle(showValues=show_x_axis)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        layout.addWidget(self.plot_widget)
        self._bars: pg.BarGraphItem | None = None
        self._bands: list[QGraphicsRectItem] = []


class StepCharts(QWidget):
    frame_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        legend = QHBoxLayout()
        for name, color in (("left", _LEFT), ("right", _RIGHT)):
            legend.addWidget(theme.make_swatch(color))
            label = QLabel(name)
            label.setStyleSheet(f"color: {theme.hexs(color)};")
            legend.addWidget(label)
        legend.addSpacing(10)
        legend.addWidget(theme.make_chip("good/minor/major", theme.ACCENT, filled=False))
        legend.addStretch(1)
        self._hint = QLabel("click a bar to view that step")
        self._hint.setStyleSheet(f"color: {theme.hexs(theme.TEXT_MUTED)};")
        legend.addWidget(self._hint)
        layout.addLayout(legend)

        self._charts: dict[str, _StepPlot] = {}
        for i, key in enumerate(("contact ms", "step length", "speed")):
            chart = _StepPlot(key, show_x_axis=(i == 2))
            chart.caption.setText(_CAPTIONS[key])
            if i > 0:
                chart.plot.setXLink(self._charts["contact ms"].plot)
            chart.plot_widget.scene().sigMouseClicked.connect(
                lambda ev, k=key: self._on_click(k, ev))
            chart.plot_widget.scene().sigMouseMoved.connect(
                lambda pos, k=key: self._on_hover(k, pos))
            layout.addWidget(chart)
            self._charts[key] = chart
        self._charts["speed"].plot.setLabel("bottom", "step #")
        self._steps: list[StepRecord] = []

    def set_steps(self, steps: list[StepRecord], length_unit: str,
                 buckets: dict[str, PhaseBucket] | None = None,
                 level: str = "trained") -> None:
        self._steps = steps
        self._charts["step length"].plot.setLabel("left", f"step length ({length_unit})")
        self._charts["speed"].plot.setLabel("left", f"speed ({length_unit}/s)")

        for chart in self._charts.values():
            chart.plot.clear()
            for band in chart._bands:
                chart.plot.removeItem(band)
            chart._bands = []
        if not steps:
            return

        frame_phase: dict[int, str] = {}
        if buckets:
            for phase, bucket in buckets.items():
                for frame in bucket.strike_frames:
                    frame_phase[frame] = phase
        checks_by_phase = sprint_checks(level) if buckets else {}

        x = np.arange(1, len(steps) + 1)
        leg_pens = [pg.mkPen(_LEFT if s.side == "left" else _RIGHT, width=2)
                   for s in steps]

        # Contact ms: severity fill + target-range zone per phase.
        contact_ms = [s.contact_time_s * 1000 for s in steps]
        contact_heights = np.array([v if np.isfinite(v) else 0.0 for v in contact_ms])
        contact_brushes = []
        for step, value in zip(steps, contact_ms):
            phase = frame_phase.get(step.strike_frame)
            check = dict(checks_by_phase.get(phase, [])).get("contact_ms") if phase else None
            finding = _evaluate(check, value / 1000, phase, step.strike_frame) \
                if check is not None else None
            color = theme.SEVERITY_COLORS[finding.severity] if finding is not None \
                else (_LEFT if step.side == "left" else _RIGHT)
            contact_brushes.append(pg.mkBrush(*color, 200))
        contact_plot = self._charts["contact ms"].plot
        contact_plot.addItem(pg.BarGraphItem(x=x, height=contact_heights, width=0.72,
                                             brushes=contact_brushes, pens=leg_pens))
        contact_plot.setXRange(0.3, len(steps) + 0.7, padding=0)
        for start_idx, end_idx, phase in _phase_spans_by_step_index(steps, frame_phase):
            check = dict(checks_by_phase.get(phase, [])).get("contact_ms")
            if check is None:
                continue
            lo, hi = check.lo * 1000, check.hi * 1000
            rect = QGraphicsRectItem(QRectF(start_idx + 1 - 0.5, lo,
                                            end_idx - start_idx + 1, hi - lo))
            rect.setBrush(QBrush(theme.qcolor(theme.GOOD, 35)))
            rect.setPen(QPen(theme.qcolor(theme.GOOD, 90), 0, Qt.PenStyle.DashLine))
            rect.setZValue(-10)
            contact_plot.addItem(rect, ignoreBounds=True)
            self._charts["contact ms"]._bands.append(rect)

        # Step length / speed: leg-colored only, no fabricated target band.
        for key, vals in (("step length", [s.step_length for s in steps]),
                          ("speed", [s.step_speed for s in steps])):
            heights = np.array([v if np.isfinite(v) else 0.0 for v in vals])
            brushes = [pg.mkBrush(*(_LEFT if s.side == "left" else _RIGHT), 200)
                      for s in steps]
            self._charts[key].plot.addItem(
                pg.BarGraphItem(x=x, height=heights, width=0.72, brushes=brushes))
            self._charts[key].plot.setXRange(0.3, len(steps) + 0.7, padding=0)

    def _on_click(self, key: str, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton or not self._steps:
            return
        plot = self._charts[key].plot
        if not plot.vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = plot.vb.mapSceneToView(ev.scenePos()).x()
        idx = int(round(x)) - 1
        if 0 <= idx < len(self._steps):
            self.frame_requested.emit(self._steps[idx].strike_frame)

    def _on_hover(self, key: str, scene_pos) -> None:
        plot = self._charts[key].plot
        if not self._steps or not plot.vb.sceneBoundingRect().contains(scene_pos):
            QToolTip.hideText()
            return
        x = plot.vb.mapSceneToView(scene_pos).x()
        idx = int(round(x)) - 1
        if not (0 <= idx < len(self._steps)):
            QToolTip.hideText()
            return
        step = self._steps[idx]
        text = {
            "contact ms": f"{step.contact_time_s * 1000:.0f} ms",
            "step length": f"{step.step_length:.2f}",
            "speed": f"{step.step_speed:.2f}",
        }[key]
        QToolTip.showText(QCursor.pos(), f"Step {idx + 1} ({step.side}): {text}")
