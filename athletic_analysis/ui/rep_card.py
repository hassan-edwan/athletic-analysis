"""Summary-first 'Rep Card': hero numbers, form score, top issues.

Answers the coach's first questions at a glance — how fast/high, and what's
the #1 thing to fix — with links into the video for proof.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)

from athletic_analysis.core.coaching import (SEVERITY_ORDER, FormFinding,
                                             summarize)
from athletic_analysis.core.confidence import ClipQuality
from athletic_analysis.core.metrics.jump import JumpMetrics
from athletic_analysis.core.metrics.sprint import SprintMetrics
from athletic_analysis.core.radar import SprintRadar
from athletic_analysis.ui.radar_widget import RadarWidget

_SEV_ICON = {"major": "⚠", "minor": "○"}  # ⚠ / ○
_CONF_COLOR = {"High": "#3da35d", "Medium": "#c9971a", "Low": "#d0453c"}


def _conf_badge(level: str, limiter: str = "") -> str:
    """Small colored HTML chip for a confidence level."""
    color = _CONF_COLOR.get(level, "#888")
    text = level if not limiter else f"{level} · {limiter}"
    return (f"<span style='color:{color}; font-size:10px;'>● {text}</span>")


def _fmt(value: float, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "–"
    return f"{value:.{decimals}f}"


class _HeroTile(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        self.value_label = QLabel("–")
        self.value_label.setStyleSheet("font-size: 22px; font-weight: 600;")
        self.caption_label = QLabel("")
        self.caption_label.setStyleSheet("color: gray; font-size: 11px;")
        self.badge_label = QLabel("")
        self.badge_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.value_label)
        layout.addWidget(self.caption_label)
        layout.addWidget(self.badge_label)

    def set(self, value: str, caption: str,
            conf: tuple[str, str] | None = None) -> None:
        self.value_label.setText(value)
        self.caption_label.setText(caption)
        if conf and conf[0]:
            self.badge_label.setText(_conf_badge(conf[0], conf[1]))
        else:
            self.badge_label.setText("")


def _hero_conf(kind: str, q: ClipQuality | None) -> tuple[str, str]:
    """Coarse per-hero confidence from clip quality, limiter chosen by metric
    kind. Honest without re-deriving full per-metric stats in the card."""
    if q is None:
        return ("", "")
    level, limiter = q.level, ""
    if kind == "timing" and not q.fps_adequate:
        level = "Medium" if level == "High" else level
        limiter = "frame rate"
    if kind in ("distance", "speed") and not q.calibrated:
        level = "Medium" if level == "High" else level
        limiter = "not calibrated"
    if q.detection_rate < 0.8 and level == "High":
        level, limiter = "Medium", "tracking"
    return (level, limiter)


class RepCard(QWidget):
    frame_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._title = QLabel("Open a video and run pose analysis.")
        self._title.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(self._title)

        self._tiles = [_HeroTile() for _ in range(4)]
        grid = QGridLayout()
        for i, tile in enumerate(self._tiles):
            grid.addWidget(tile, i // 2, i % 2)
        layout.addLayout(grid)

        self._radar = RadarWidget()
        layout.addWidget(self._radar)

        self._score = QLabel("")
        self._score.setWordWrap(True)
        layout.addWidget(self._score)

        self._quality = QLabel("")
        self._quality.setWordWrap(True)
        self._quality.setTextFormat(Qt.TextFormat.RichText)
        self._quality.setStyleSheet("font-size: 11px;")
        layout.addWidget(self._quality)

        self._issues_box = QVBoxLayout()
        layout.addLayout(self._issues_box)
        layout.addStretch(1)
        self._issue_buttons: list[QPushButton] = []

    def _set_issues(self, findings: list[FormFinding]) -> None:
        for btn in self._issue_buttons:
            self._issues_box.removeWidget(btn)
            btn.deleteLater()
        self._issue_buttons = []
        # Prefer confident faults as headline issues; low-confidence ones sink
        # to the bottom and are tagged, never silently trusted.
        faults = [f for f in findings if f.severity != "good"]

        def conf_rank(f: FormFinding) -> int:
            lvl = f.confidence.level if f.confidence else "High"
            return {"High": 0, "Medium": 1, "Low": 2}.get(lvl, 0)

        faults.sort(key=lambda f: (conf_rank(f), SEVERITY_ORDER.get(f.severity, 3)))
        problems = faults[:3]
        if not problems and findings:
            label = QPushButton("✓ No form faults detected in this rep")
            label.setFlat(True)
            label.setStyleSheet("text-align: left; color: #3da35d; border: none;")
            self._issues_box.addWidget(label)
            self._issue_buttons.append(label)
            return
        for f in problems:
            icon = _SEV_ICON.get(f.severity, "")
            color = "#d0453c" if f.severity == "major" else "#c9971a"
            low = f.confidence is not None and f.confidence.level == "Low"
            tag = "  ⚠ low confidence — verify" if low else ""
            btn = QPushButton(f"{icon} {f.metric} ({f.phase}): {f.value_text} "
                              f"vs optimal {f.target_text} — click to view{tag}")
            btn.setFlat(True)
            tip = f.cue + (f"\n\nSource: {f.source}" if f.source else "")
            if f.confidence and f.confidence.limiter:
                tip += f"\nConfidence limited by: {f.confidence.limiter}"
            btn.setToolTip(tip)
            btn.setStyleSheet(
                f"text-align: left; color: {color}; border: none; padding: 2px;")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, fr=f.frame: self.frame_requested.emit(fr))
            self._issues_box.addWidget(btn)
            self._issue_buttons.append(btn)

    def _set_quality(self, q: ClipQuality | None, model_tier: str) -> None:
        if q is None:
            self._quality.setText("")
            return
        badge = _conf_badge(q.level)
        notes = " · ".join(q.notes)
        self._quality.setText(
            f"Analysis quality: {badge}<br>"
            f"<span style='color:gray;'>{notes} · model: {model_tier}</span>")

    def show_sprint(self, m: SprintMetrics | None,
                    findings: list[FormFinding],
                    quality: ClipQuality | None = None,
                    model_tier: str = "Balanced",
                    radar: SprintRadar | None = None) -> None:
        if m is None or not m.steps:
            self.clear("No steps detected — run pose analysis on a sprint clip.")
            return
        self._radar.set_radar(radar)
        unit = m.length_unit
        self._title.setText("Sprint")
        self._tiles[0].set(f"{_fmt(m.max_speed)} {unit}/s", "top speed",
                           _hero_conf("speed", quality))
        self._tiles[1].set(f"{_fmt(m.cadence_spm, 0)}", "steps / min",
                           _hero_conf("timing", quality))
        self._tiles[2].set(f"{_fmt(m.mean_contact_s * 1000, 0)} ms",
                           "avg ground contact", _hero_conf("timing", quality))
        self._tiles[3].set(f"{_fmt(m.mean_step_length)} {unit}", "avg step length",
                           _hero_conf("distance", quality))
        self._score.setText(summarize(findings))
        self._set_quality(quality, model_tier)
        self._set_issues(findings)

    def show_jump(self, m: JumpMetrics | None,
                  findings: list[FormFinding],
                  quality: ClipQuality | None = None,
                  model_tier: str = "Balanced") -> None:
        if m is None or m.takeoff_frame < 0:
            self.clear("No jump detected — run pose analysis on a jump clip.")
            return
        self._radar.set_radar(None)
        self._title.setText("Jump")
        self._tiles[0].set(f"{_fmt(m.jump_height_flight_m)} m",
                           "jump height (flight time)", _hero_conf("timing", quality))
        self._tiles[1].set(f"{_fmt(m.takeoff_velocity_m_s)} m/s", "takeoff velocity",
                           _hero_conf("timing", quality))
        self._tiles[2].set(f"{_fmt(m.flight_time_s * 1000, 0)} ms", "flight time",
                           _hero_conf("timing", quality))
        self._tiles[3].set(f"{_fmt(m.countermovement_depth)} {m.length_unit}",
                           "countermovement depth", _hero_conf("distance", quality))
        self._score.setText(summarize(findings))
        self._set_quality(quality, model_tier)
        self._set_issues(findings)

    def clear(self, message: str) -> None:
        self._radar.set_radar(None)
        self._title.setText(message)
        for tile in self._tiles:
            tile.set("–", "")
        self._score.setText("")
        self._quality.setText("")
        self._set_issues([])
