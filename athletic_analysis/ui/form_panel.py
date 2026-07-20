"""Form-analysis panel: severity-colored coaching findings per phase, plus a
root-cause pane (technical causes / muscle factors / drills) for the selected
fault."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QLabel, QTableWidget, QTableWidgetItem,
                               QTextBrowser, QVBoxLayout, QWidget)

from athletic_analysis.core.coaching import FormFinding, summarize
from athletic_analysis.core.diagnostics import Diagnosis, diagnose

_SEVERITY_BG = {
    "good": QColor(60, 170, 90, 55),
    "minor": QColor(235, 180, 40, 70),
    "major": QColor(220, 70, 60, 80),
}
_SEVERITY_LABEL = {"good": "OK", "minor": "Minor", "major": "Fix"}
_CONF_FG = {"High": QColor(80, 200, 120), "Medium": QColor(225, 175, 55),
            "Low": QColor(225, 90, 80)}


class FormPanel(QWidget):
    frame_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._summary = QLabel("Run pose analysis to get form feedback.")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setWordWrap(True)
        self._table.cellClicked.connect(self._on_cell_clicked)
        self._table.currentCellChanged.connect(self._on_row_changed)
        layout.addWidget(self._table)
        self._detail = QTextBrowser()
        self._detail.setOpenExternalLinks(False)
        self._detail.setMaximumHeight(170)
        self._detail.hide()
        layout.addWidget(self._detail)
        self._row_frames: list[int] = []
        self._findings: list[FormFinding] = []

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._row_frames):
            self.frame_requested.emit(self._row_frames[row])

    def _on_row_changed(self, row: int, _col: int = 0,
                        _prev_row: int = -1, _prev_col: int = -1) -> None:
        if not (0 <= row < len(self._findings)):
            self._detail.hide()
            return
        finding = self._findings[row]
        diag = diagnose(finding)
        if diag is None:
            if finding.severity == "good" or finding.phase == "jump":
                self._detail.hide()
            else:
                self._detail.setHtml(
                    "<i>No root-cause entry for this deviation — it usually "
                    "indicates a tracking or capture-FPS issue rather than a "
                    "form fault. Verify the keypoints on the flagged frames."
                    "</i>")
                self._detail.show()
            return
        self._detail.setHtml(self._diagnosis_html(finding, diag))
        self._detail.show()

    @staticmethod
    def _diagnosis_html(finding: FormFinding, diag: Diagnosis) -> str:
        def bullets(items: tuple[str, ...]) -> str:
            lis = "".join(f"<li>{item}</li>" for item in items)
            return f"<ul style='margin: 2px 0 6px 16px;'>{lis}</ul>"

        color = "#d0453c" if finding.severity == "major" else "#c9971a"
        parts = [f"<b style='color:{color};'>{diag.title}</b>"
                 f" <span style='color:gray;'>({finding.phase} · measured "
                 f"{finding.value_text} vs optimal {finding.target_text})</span>",
                 "<b>Why this happens</b>", bullets(diag.technical_causes),
                 "<b>Likely physical limiters</b>", bullets(diag.muscle_factors),
                 "<b>Corrective drills</b>", bullets(diag.drills)]
        if diag.phase_note:
            parts.append(f"<i style='color:gray;'>{diag.phase_note}</i>")
        if diag.source:
            parts.append(f"<span style='color:gray; font-size:10px;'>Source: "
                         f"{diag.source}</span>")
        return "<br>".join(parts)

    def show_findings(self, findings: list[FormFinding]) -> None:
        self._summary.setText(summarize(findings))
        headers = ["", "Phase", "Metric", "Measured", "Optimal", "Conf.",
                   "Coaching cue"]
        self._table.clear()
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(findings))
        self._row_frames = [f.frame for f in findings]
        self._findings = findings
        self._detail.hide()
        for r, f in enumerate(findings):
            conf_level = f.confidence.level if f.confidence else "–"
            conf_text = conf_level
            if f.confidence and f.confidence.limiter:
                conf_text = f"{conf_level} · {f.confidence.limiter}"
            cells = [_SEVERITY_LABEL[f.severity], f.phase, f.metric,
                     f.value_text, f.target_text, conf_text, f.cue]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setBackground(_SEVERITY_BG[f.severity])
                if c == 5 and f.confidence:  # confidence column colored
                    item.setForeground(_CONF_FG.get(f.confidence.level,
                                                    QColor(180, 180, 180)))
                if f.source:
                    item.setToolTip(f"Source: {f.source}")
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()
        # Keep the cue column readable instead of one endless line.
        cue_col = len(headers) - 1
        if self._table.columnWidth(cue_col) > 420:
            self._table.setColumnWidth(cue_col, 420)
        self._table.resizeRowsToContents()
