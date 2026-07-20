"""Video-suitability panel: grade, itemized issues, and preprocessing toggles.

Recommended transforms are pre-checked; the user can override before running
analysis. Emits `transforms_changed` so the main window keeps the effective
Preprocessor in sync.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel,
                               QVBoxLayout, QWidget)

from athletic_analysis.core.assessment import VideoAssessment
from athletic_analysis.ui import theme

_GRADE_COLOR = {k: theme.hexs(v) for k, v in theme.GRADE_COLORS.items()}
_SEV_COLOR = {"minor": theme.hexs(theme.WARN), "major": theme.hexs(theme.BAD)}
_TRANSFORM_LABELS = {
    "reframe": "Auto-reframe (crop + upscale athlete)",
    "enhance": "Enhance contrast / brightness",
    "deinterlace": "Deinterlace",
}


class SuitabilityPanel(QWidget):
    transforms_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._grade = QLabel("Open a video to assess it.")
        self._grade.setTextFormat(Qt.TextFormat.RichText)
        self._grade.setWordWrap(True)
        layout.addWidget(self._grade)

        self._issues = QVBoxLayout()
        layout.addLayout(self._issues)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)

        layout.addWidget(QLabel("<b>Preprocessing for analysis</b>"))
        self._rotate_note = QLabel("")
        self._rotate_note.setWordWrap(True)
        layout.addWidget(self._rotate_note)
        self._checks: dict[str, QCheckBox] = {}
        for key, label in _TRANSFORM_LABELS.items():
            cb = QCheckBox(label)
            cb.toggled.connect(lambda _=False: self.transforms_changed.emit())
            self._checks[key] = cb
            layout.addWidget(cb)
        layout.addStretch(1)
        self._issue_widgets: list[QWidget] = []
        self._assessment: VideoAssessment | None = None

    @property
    def assessment(self) -> VideoAssessment | None:
        return self._assessment

    def set_busy(self) -> None:
        self._grade.setText("Assessing video suitability…")

    def selected_transforms(self) -> dict[str, bool]:
        return {k: cb.isChecked() for k, cb in self._checks.items()}

    def rotation(self) -> int:
        return self._assessment.rotation_suggestion if self._assessment else 0

    def clear(self) -> None:
        self._assessment = None
        self._grade.setText("Open a video to assess it.")
        self._clear_issues()
        for cb in self._checks.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self._rotate_note.setText("")

    def _clear_issues(self) -> None:
        for w in self._issue_widgets:
            self._issues.removeWidget(w)
            w.deleteLater()
        self._issue_widgets = []

    def show_assessment(self, a: VideoAssessment) -> None:
        self._assessment = a
        color = _GRADE_COLOR.get(a.grade, "#888")
        self._grade.setText(
            f"<b>Suitability: <span style='color:{color}'>{a.grade}</span></b> "
            f"· {a.fps:.0f} fps · {a.width}×{a.height}")
        self._clear_issues()
        if not a.issues:
            row = QWidget()
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(theme.make_chip("Good", theme.GOOD))
            ok = QLabel("No problems detected — good to analyze.")
            ok.setStyleSheet(f"color: {theme.hexs(theme.GOOD)};")
            hl.addWidget(ok)
            hl.addStretch(1)
            self._issues.addWidget(row)
            self._issue_widgets.append(row)
        for issue in a.issues:
            c = _SEV_COLOR.get(issue.severity, "#888")
            row = QWidget()
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(theme.make_chip(issue.severity, theme.SEVERITY_COLORS.get(
                issue.severity, theme.WARN)))
            text = QLabel(f"<b style='color:{c}'>{issue.title}</b><br>"
                         f"<span style='color:{theme.hexs(theme.TEXT_MUTED)}'>{issue.detail}</span>")
            text.setTextFormat(Qt.TextFormat.RichText)
            text.setWordWrap(True)
            hl.addWidget(text, stretch=1)
            self._issues.addWidget(row)
            self._issue_widgets.append(row)

        # Pre-check recommended transforms.
        rec = set(a.recommended_transforms)
        for key, cb in self._checks.items():
            cb.blockSignals(True)
            cb.setChecked(key in rec)
            cb.blockSignals(False)
        if a.rotation_suggestion:
            self._rotate_note.setText(
                f"Auto-rotate {a.rotation_suggestion}° applied to correct "
                "orientation.")
        else:
            self._rotate_note.setText("")
        self.transforms_changed.emit()
