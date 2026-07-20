"""Per-step bar charts: contact time, step length, step speed — colored by leg
so left/right asymmetry and step-to-step trends are visible at a glance.
Clicking a bar seeks the video to that foot strike."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from athletic_analysis.core.metrics.sprint import StepRecord

_LEFT = (80, 200, 80)
_RIGHT = (60, 140, 255)


class StepCharts(QWidget):
    frame_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        legend = QHBoxLayout()
        for name, color in (("left", _LEFT), ("right", _RIGHT)):
            chip = QLabel(f"■ {name}")
            chip.setStyleSheet("color: rgb({},{},{});".format(*color))
            legend.addWidget(chip)
        legend.addStretch(1)
        self._hint = QLabel("click a bar to view that step")
        self._hint.setStyleSheet("color: gray;")
        legend.addWidget(self._hint)
        layout.addLayout(legend)

        self._layout_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self._layout_widget)
        self._plots: list[pg.PlotItem] = []
        for row, label in enumerate(("contact ms", "step length", "speed")):
            plot = self._layout_widget.addPlot(row=row, col=0)
            plot.setLabel("left", label)
            plot.getAxis("bottom").setStyle(showValues=(row == 2))
            plot.setMouseEnabled(x=False, y=False)
            plot.hideButtons()
            if row > 0:
                plot.setXLink(self._plots[0])
            self._plots.append(plot)
        self._plots[2].setLabel("bottom", "step #")
        self._steps: list[StepRecord] = []
        self._layout_widget.scene().sigMouseClicked.connect(self._on_click)

    def set_steps(self, steps: list[StepRecord], length_unit: str) -> None:
        self._steps = steps
        self._plots[1].setLabel("left", f"step length ({length_unit})")
        self._plots[2].setLabel("left", f"speed ({length_unit}/s)")
        values = [
            [s.contact_time_s * 1000 for s in steps],
            [s.step_length for s in steps],
            [s.step_speed for s in steps],
        ]
        brushes = [pg.mkBrush(*(_LEFT if s.side == "left" else _RIGHT), 200)
                   for s in steps]
        x = np.arange(1, len(steps) + 1)
        for plot, vals in zip(self._plots, values):
            plot.clear()
            if not steps:
                continue
            heights = np.array([v if np.isfinite(v) else 0.0 for v in vals])
            bars = pg.BarGraphItem(x=x, height=heights, width=0.72, brushes=brushes)
            plot.addItem(bars)
            plot.setXRange(0.3, len(steps) + 0.7, padding=0)

    def _on_click(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton or not self._steps:
            return
        for plot in self._plots:
            if plot.vb.sceneBoundingRect().contains(ev.scenePos()):
                x = plot.vb.mapSceneToView(ev.scenePos()).x()
                idx = int(round(x)) - 1
                if 0 <= idx < len(self._steps):
                    self.frame_requested.emit(self._steps[idx].strike_frame)
                return
