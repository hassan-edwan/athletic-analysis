"""Pentagon (radar) chart of the five sprint-mechanics factor scores.

Custom QPainter widget: the chart is a small, fixed-geometry summary glyph
(no zoom/pan/linking), so plain painting gives crisper labels and simpler
hover handling than forcing pyqtgraph into polar duty.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from athletic_analysis.core.radar import SprintRadar
from athletic_analysis.ui import theme

_GRID = QColor(128, 128, 128, 60)
_SPOKE = QColor(128, 128, 128, 90)
_FILL = theme.qcolor(theme.ACCENT, 70)
_OUTLINE = theme.qcolor(theme.ACCENT, 220)
_LABEL = QColor(160, 160, 160)
_NA = QColor(120, 120, 120)
_SCORE_COLORS = ((60, theme.qcolor(theme.BAD)),    # < 60: major territory
                 (85, theme.qcolor(theme.WARN)),   # 60–85: needs work
                 (101, theme.qcolor(theme.GOOD)))  # >= 85: solid

# Short labels, same order as core.radar.RADAR_AXES.
_SHORT = ("Stiffness", "Front-side", "Posture", "Foot placement", "Rhythm")


def _score_color(score: float) -> QColor:
    if not np.isfinite(score):
        return _NA
    for hi, color in _SCORE_COLORS:
        if score < hi:
            return color
    return _SCORE_COLORS[-1][1]


class RadarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._radar: SprintRadar | None = None
        self._vertices: list[QPointF] = []  # outer (r=100) vertex per axis
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(230)
        self.setVisible(False)

    def set_radar(self, radar: SprintRadar | None) -> None:
        self._radar = radar
        self.setVisible(radar is not None)
        self.update()

    # --- geometry ----------------------------------------------------------

    def _layout(self) -> tuple[QPointF, float]:
        w, h = self.width(), self.height()
        center = QPointF(w / 2, h / 2 + 6)  # nudge down; top label needs room
        radius = max(10.0, min(w / 2 - 70, h / 2 - 26))
        return center, radius

    @staticmethod
    def _point(center: QPointF, radius: float, i: int, r: float) -> QPointF:
        # Vertex 0 at 12 o'clock, clockwise; screen y grows downward.
        angle = -math.pi / 2 + 2 * math.pi * i / 5
        return QPointF(center.x() + radius * (r / 100.0) * math.cos(angle),
                       center.y() + radius * (r / 100.0) * math.sin(angle))

    # --- painting ----------------------------------------------------------

    def paintEvent(self, _event) -> None:
        if self._radar is None or len(self._radar.axes) != 5:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center, radius = self._layout()
        self._vertices = [self._point(center, radius, i, 100) for i in range(5)]

        # Grid pentagons + spokes.
        painter.setPen(QPen(_GRID, 1))
        for ring in (20, 40, 60, 80, 100):
            poly = QPolygonF([self._point(center, radius, i, ring)
                              for i in range(5)])
            painter.drawPolygon(poly)
        painter.setPen(QPen(_SPOKE, 1))
        for vertex in self._vertices:
            painter.drawLine(center, vertex)

        # Score polygon (NaN axes collapse to the center).
        scores = [a.score for a in self._radar.axes]
        pts = [self._point(center, radius, i,
                           s if np.isfinite(s) else 0.0)
               for i, s in enumerate(scores)]
        painter.setPen(QPen(_OUTLINE, 2))
        painter.setBrush(_FILL)
        painter.drawPolygon(QPolygonF(pts))
        painter.setBrush(_OUTLINE)
        for pt, s in zip(pts, scores):
            if np.isfinite(s):
                painter.drawEllipse(pt, 3, 3)

        # Axis labels + numeric scores just outside each vertex.
        font = QFont(self.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 1))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        for i, axis in enumerate(self._radar.axes):
            score_txt = (f"{axis.score:.0f}" if np.isfinite(axis.score)
                         else "n/a")
            text = f"{_SHORT[i]} {score_txt}"
            anchor = self._point(center, radius, i, 116)
            tw = metrics.horizontalAdvance(text)
            x = anchor.x() - tw / 2
            if anchor.x() < center.x() - radius * 0.3:
                x = anchor.x() - tw + 8
            elif anchor.x() > center.x() + radius * 0.3:
                x = anchor.x() - 8
            painter.setPen(_LABEL)
            name_w = metrics.horizontalAdvance(_SHORT[i] + " ")
            painter.drawText(QPointF(x, anchor.y() + 4), _SHORT[i] + " ")
            painter.setPen(_score_color(axis.score))
            painter.drawText(QPointF(x + name_w, anchor.y() + 4), score_txt)

        # Overall in the center.
        if np.isfinite(self._radar.overall):
            painter.setPen(_LABEL)
            text = f"overall {self._radar.overall:.0f}"
            tw = painter.fontMetrics().horizontalAdvance(text)
            painter.drawText(QPointF(center.x() - tw / 2, center.y() + 4), text)
        painter.end()

    # --- hover tooltips ----------------------------------------------------

    def mouseMoveEvent(self, event) -> None:
        if self._radar is None or not self._vertices:
            return
        pos = event.position()
        dists = [math.hypot(pos.x() - v.x(), pos.y() - v.y())
                 for v in self._vertices]
        i = int(np.argmin(dists))
        if dists[i] < 60:
            axis = self._radar.axes[i]
            extra = f"\nfrom {axis.n_steps} steps" if axis.n_steps else ""
            QToolTip.showText(event.globalPosition().toPoint(),
                              f"{axis.name}: {axis.detail}{extra}", self)
        else:
            QToolTip.hideText()
