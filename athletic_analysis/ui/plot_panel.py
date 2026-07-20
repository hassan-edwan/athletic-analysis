"""Angle and velocity curves vs. time (pyqtgraph) with a cursor synced to the
video frame. Angles use the left axis (deg), velocities the right axis."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGraphicsRectItem,
                               QHBoxLayout, QLabel, QVBoxLayout, QWidget)

from athletic_analysis.ui import theme
from athletic_analysis.ui.theme import PHASE_COLORS

# (key, label, color, is_velocity). Left/right pairs use theme.LEG_LEFT/
# LEG_RIGHT (and their light/dark shades) so a curve's color always matches
# which leg it's tracking on the video overlay — see theme.py's docstring for
# the BGR/RGB mixup this used to have (right-side curves rendered blue-ish).
_SERIES = [
    ("knee_l", "Knee L", theme.LEG_LEFT, False),
    ("knee_r", "Knee R", theme.LEG_RIGHT, False),
    ("hip_l", "Hip L", theme.LEG_LEFT_LIGHT, False),
    ("hip_r", "Hip R", theme.LEG_RIGHT_LIGHT, False),
    ("ankle_l", "Ankle L", theme.LEG_LEFT_DARK, False),
    ("ankle_r", "Ankle R", theme.LEG_RIGHT_DARK, False),
    ("trunk_lean", "Trunk lean", (230, 200, 160), False),
    ("thigh_l", "Thigh L", (200, 230, 90), False),
    ("thigh_r", "Thigh R", (235, 165, 90), False),
    ("run_speed", "Run speed", (255, 210, 60), True),
    ("hip_speed", "Hip speed", (255, 170, 40), True),
    ("hip_vx", "Hip horiz vel", (255, 140, 60), True),
    ("hip_vy", "Hip vert vel", (255, 90, 160), True),
]
_DEFAULT_ON = {"knee_l", "knee_r", "trunk_lean", "run_speed"}

_PRESETS: dict[str, set[str] | None] = {
    "Speed": {"run_speed", "hip_vy"},
    "Ground contact": {"knee_l", "knee_r", "ankle_l", "ankle_r"},
    "Posture": {"trunk_lean", "thigh_l", "thigh_r"},
    "Custom": None,
}
_PRESET_CAPTIONS = {
    "Speed": "The two clearest markers of top-end running speed.",
    "Ground contact": "How the legs behave right around each foot strike.",
    "Posture": "Trunk and thigh angles — the shape of your upper body and "
               "front-side mechanics.",
    "Custom": "Pick any combination of curves below.",
}

# One line per curve — what it's showing, not a verdict on it (that's what
# the phase target bands drawn by set_phases() are for). Reused as each
# checkbox's tooltip so a curve name is never the only explanation offered.
_SERIES_HELP = {
    "knee_l": "Left knee angle over time — sharpest dips happen at ground contact.",
    "knee_r": "Right knee angle over time — sharpest dips happen at ground contact.",
    "hip_l": "Left hip angle (trunk-to-thigh) over time.",
    "hip_r": "Right hip angle (trunk-to-thigh) over time.",
    "ankle_l": "Left ankle angle over time — shows push-off timing.",
    "ankle_r": "Right ankle angle over time — shows push-off timing.",
    "trunk_lean": "How far forward your torso tips, over time — a big lean "
                  "out of the start, close to upright at top speed.",
    "thigh_l": "Left thigh angle vs. vertical over time — how high the "
               "knee drives at each instant.",
    "thigh_r": "Right thigh angle vs. vertical over time — how high the "
               "knee drives at each instant.",
    "run_speed": "Your ~0.4 s-smoothed running speed — the clearest single "
                 "curve for seeing where in the rep you were fastest.",
    "hip_speed": "Raw hip speed — noisier than run speed, useful for "
                 "spotting individual bursts.",
    "hip_vx": "Horizontal hip velocity — the raw component run speed is built from.",
    "hip_vy": "Vertical hip velocity — bounce/oscillation in your stride.",
}


class PlotPanel(QWidget):
    frame_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fps = 30.0
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._vel_curves: dict[str, pg.PlotDataItem] = {}
        self._data: dict[str, np.ndarray] = {}
        self._vel_unit = "BH/s"
        self._phase_regions: list = []
        self._phase_labels: list[tuple[pg.TextItem, float]] = []
        self._band_items: dict[str, list[QGraphicsRectItem]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        checks = QHBoxLayout()
        self._preset_combo = QComboBox()
        for name in _PRESETS:
            self._preset_combo.addItem(name)
        self._preset_combo.setCurrentText("Custom")
        self._preset_combo.currentTextChanged.connect(self._apply_preset)
        checks.addWidget(QLabel("View:"))
        checks.addWidget(self._preset_combo)
        self._checks: dict[str, QCheckBox] = {}
        for key, label, _color, _vel in _SERIES:
            cb = QCheckBox(label)
            cb.setChecked(key in _DEFAULT_ON)
            cb.setToolTip(_SERIES_HELP.get(key, ""))
            cb.toggled.connect(self._on_check_toggled)
            self._checks[key] = cb
            checks.addWidget(cb)
        checks.addStretch(1)
        layout.addLayout(checks)

        self._preset_caption = QLabel(_PRESET_CAPTIONS["Custom"])
        self._preset_caption.setStyleSheet(
            f"color: {theme.hexs(theme.TEXT_MUTED)}; font-size: 10px;")
        layout.addWidget(self._preset_caption)

        self._plot = pg.PlotWidget()
        item = self._plot.getPlotItem()
        item.setLabel("bottom", "time", units="s")
        item.setLabel("left", "angle (deg)")
        item.addLegend(offset=(10, 10))

        # Second ViewBox so velocities get their own y-scale on the right axis.
        self._vel_vb = pg.ViewBox()
        item.showAxis("right")
        item.scene().addItem(self._vel_vb)
        item.getAxis("right").linkToView(self._vel_vb)
        self._vel_vb.setXLink(item.vb)
        item.vb.sigResized.connect(self._sync_vel_vb)
        item.vb.sigYRangeChanged.connect(self._reposition_phase_labels)
        self._sync_vel_vb()

        self._cursor = pg.InfiniteLine(angle=90, movable=True,
                                       pen=pg.mkPen((240, 240, 240), width=1))
        self._cursor.sigPositionChanged.connect(self._on_cursor_moved)
        self._plot.addItem(self._cursor)
        self._plot.scene().sigMouseClicked.connect(self._on_click)

        # Hover crosshair + readout of every visible curve at the cursor.
        self._hover_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen((150, 150, 150), width=1, style=Qt.PenStyle.DotLine))
        self._hover_line.hide()
        self._plot.addItem(self._hover_line, ignoreBounds=True)
        self._hover_text = pg.TextItem(anchor=(0, 0),
                                       fill=pg.mkBrush(20, 20, 24, 200),
                                       border=pg.mkPen(90, 90, 100))
        self._hover_text.setZValue(100)
        self._hover_text.hide()
        self._plot.addItem(self._hover_text, ignoreBounds=True)
        self._plot.scene().sigMouseMoved.connect(self._on_hover)

        layout.addWidget(self._plot)
        self._updating_cursor = False

    def _sync_vel_vb(self) -> None:
        vb = self._plot.getPlotItem().vb
        self._vel_vb.setGeometry(vb.sceneBoundingRect())
        self._vel_vb.linkedViewChanged(vb, self._vel_vb.XAxis)

    def set_data(self, angles: dict[str, np.ndarray],
                 velocities: dict[str, np.ndarray], fps: float,
                 velocity_unit: str = "BH/s") -> None:
        self._fps = fps
        self._data = {**angles, **velocities}
        self._vel_unit = velocity_unit
        item = self._plot.getPlotItem()
        for curve in self._curves.values():
            item.removeItem(curve)
        for curve in self._vel_curves.values():
            self._vel_vb.removeItem(curve)
            if item.legend is not None:
                item.legend.removeItem(curve)
        self._curves = {}
        self._vel_curves = {}
        item.setLabel("right", f"velocity ({velocity_unit})")

        data = {**angles, **velocities}
        for key, label, color, is_vel in _SERIES:
            series = data.get(key)
            if series is None:
                continue
            t = np.arange(len(series)) / fps
            if is_vel:
                curve = pg.PlotDataItem(t, series, pen=pg.mkPen(color, width=1.5,
                                        style=Qt.PenStyle.DashLine),
                                        connect="finite")
                self._vel_vb.addItem(curve)
                if item.legend is not None:
                    item.legend.addItem(curve, label)
                self._vel_curves[key] = curve
            else:
                curve = item.plot(t, series, pen=pg.mkPen(color, width=1.5),
                                  name=label, connect="finite")
                self._curves[key] = curve
        self._vel_vb.enableAutoRange(pg.ViewBox.YAxis)
        self._refresh_visibility()

    def set_phases(self, spans: list[tuple[int, int, str]],
                   targets: dict[str, dict[str, tuple[float, float]]] | None = None
                   ) -> None:
        """Draw phase spans as tinted backgrounds with labels, and (optionally)
        per-phase optimal-range bands for series listed in `targets`."""
        item = self._plot.getPlotItem()
        for region in self._phase_regions:
            item.removeItem(region)
        for label, _t in self._phase_labels:
            item.removeItem(label)
        for rects in self._band_items.values():
            for rect in rects:
                item.removeItem(rect)
        self._phase_regions = []
        self._phase_labels = []
        self._band_items = {}

        for f0, f1, name in spans:
            if f1 <= f0:
                continue
            t0, t1 = f0 / self._fps, (f1 + 1) / self._fps
            color = PHASE_COLORS.get(name, (160, 160, 160))
            region = pg.LinearRegionItem(
                values=(t0, t1), movable=False,
                brush=pg.mkBrush(*color, 26), pen=pg.mkPen(*color, 60, width=1))
            region.setZValue(-20)
            item.addItem(region, ignoreBounds=True)
            self._phase_regions.append(region)
            label = pg.TextItem(name, color=pg.mkColor(*color, 200), anchor=(0.5, 1))
            label.setZValue(-10)
            item.addItem(label, ignoreBounds=True)
            self._phase_labels.append((label, (t0 + t1) / 2))

            if not targets:
                continue
            for key, phase_map in targets.items():
                if name not in phase_map or key not in self._data:
                    continue
                lo, hi = phase_map[name]
                series_color = next((c for k, _l, c, _v in _SERIES if k == key),
                                    (200, 200, 200))
                rect = QGraphicsRectItem(QRectF(t0, lo, t1 - t0, hi - lo))
                rect.setBrush(QBrush(QColor(*series_color, 38)))
                rect.setPen(QPen(QColor(*series_color, 90), 0, Qt.PenStyle.DashLine))
                rect.setZValue(-15)
                item.addItem(rect, ignoreBounds=True)
                self._band_items.setdefault(key, []).append(rect)
        self._reposition_phase_labels()
        self._refresh_visibility()

    def _apply_preset(self, name: str) -> None:
        self._preset_caption.setText(_PRESET_CAPTIONS.get(name, ""))
        wanted = _PRESETS.get(name)
        if wanted is None:  # Custom: leave checkboxes as they are
            return
        for key, cb in self._checks.items():
            cb.blockSignals(True)
            cb.setChecked(key in wanted)
            cb.blockSignals(False)
        self._refresh_visibility()

    def _on_check_toggled(self) -> None:
        # Manual toggles mean the user left the preset.
        if self._preset_combo.currentText() != "Custom":
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentText("Custom")
            self._preset_combo.blockSignals(False)
            self._preset_caption.setText(_PRESET_CAPTIONS["Custom"])
        self._refresh_visibility()

    def zoom_to_frames(self, f0: int, f1: int) -> None:
        if f1 <= f0:
            return
        self._plot.getPlotItem().vb.setXRange(f0 / self._fps, f1 / self._fps,
                                              padding=0.03)

    def _reposition_phase_labels(self) -> None:
        if not self._phase_labels:
            return
        (_x0, _x1), (y0, _y1) = self._plot.getPlotItem().vb.viewRange()
        for label, t_mid in self._phase_labels:
            label.setPos(t_mid, y0)

    def _refresh_visibility(self) -> None:
        for key, curve in {**self._curves, **self._vel_curves}.items():
            curve.setVisible(self._checks[key].isChecked())
        for key, rects in self._band_items.items():
            visible = key in self._checks and self._checks[key].isChecked()
            for rect in rects:
                rect.setVisible(visible)

    def set_current_frame(self, frame: int) -> None:
        self._updating_cursor = True
        self._cursor.setPos(frame / self._fps)
        self._updating_cursor = False

    def _on_cursor_moved(self) -> None:
        if self._updating_cursor:
            return
        self.frame_requested.emit(round(self._cursor.value() * self._fps))

    def _on_hover(self, scene_pos) -> None:
        item = self._plot.getPlotItem()
        if not self._data or not item.vb.sceneBoundingRect().contains(scene_pos):
            self._hover_line.hide()
            self._hover_text.hide()
            return
        view_pos = item.vb.mapSceneToView(scene_pos)
        t = view_pos.x()
        n = max((len(s) for s in self._data.values()), default=0)
        frame = round(t * self._fps)
        if n == 0 or frame < 0 or frame >= n:
            self._hover_line.hide()
            self._hover_text.hide()
            return
        lines = [f"<b>t = {frame / self._fps:.3f} s · frame {frame}</b>"]
        for key, label, color, is_vel in _SERIES:
            series = self._data.get(key)
            if series is None or not self._checks[key].isChecked():
                continue
            if frame >= len(series) or not np.isfinite(series[frame]):
                continue
            unit = self._vel_unit if is_vel else "°"
            hexcolor = "#{:02x}{:02x}{:02x}".format(*color)
            lines.append(f"<span style='color:{hexcolor}'>●</span> "
                         f"{label}: {series[frame]:.1f} {unit}")
        self._hover_text.setHtml("<div style='font-size:10pt'>"
                                 + "<br>".join(lines) + "</div>")
        # Anchor left of the crosshair when near the right edge.
        (x0, x1), (y0, y1) = item.vb.viewRange()
        near_right = t > x0 + 0.72 * (x1 - x0)
        self._hover_text.setAnchor((1, 0) if near_right else (0, 0))
        self._hover_text.setPos(t, y1)
        self._hover_line.setPos(t)
        self._hover_line.show()
        self._hover_text.show()

    def leaveEvent(self, event) -> None:
        self._hover_line.hide()
        self._hover_text.hide()
        super().leaveEvent(event)

    def _on_click(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        point = self._plot.getPlotItem().vb.mapSceneToView(ev.scenePos())
        self.frame_requested.emit(round(point.x() * self._fps))
