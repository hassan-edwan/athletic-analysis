"""Shared visual language for every panel: one palette, one flat dark QSS
theme, applied app-wide from `__main__.py`.

Every color used to mean something before this module existed too — phase
tints, severity/confidence colors, left/right leg colors — just copy-pasted
into five-plus files as raw tuples. Consolidating them here is not only
tidier, it fixes a real bug: `core/pose/skeleton.py` draws the video overlay
with OpenCV, which takes **BGR** tuples, while every other panel (pyqtgraph,
QColor, Qt style sheets) takes **RGB**. Several UI files copied the video
overlay's literal BGR tuple for "right" — (60, 140, 255) — straight into an
RGB context, so the right leg has been rendering as blue-ish in every chart,
table, and marker while showing correctly as orange on the video itself.
The colors below are defined once, correctly, as the human-visible RGB you'd
name (`LEG_RIGHT` = orange); `core/pose/skeleton.py` reverses them for BGR.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QStyleFactory

Rgb = tuple[int, int, int]

# --- Surfaces --------------------------------------------------------------

BACKGROUND: Rgb = (11, 12, 15)
SURFACE: Rgb = (25, 26, 32)
SURFACE_RAISED: Rgb = (34, 35, 41)
SURFACE_HOVER: Rgb = (42, 44, 51)
HAIRLINE: Rgb = (255, 255, 255)  # used with low alpha, see `qcolor(HAIRLINE, 22)`
TEXT: Rgb = (226, 227, 230)
TEXT_MUTED: Rgb = (150, 152, 158)
TEXT_DISABLED: Rgb = (110, 112, 118)

# Brand accent — a cool "timing display" blue, deliberately distinct from
# every data-semantic color below so chrome never reads as data.
ACCENT: Rgb = (59, 156, 255)

# --- Data semantics ----------------------------------------------------

GOOD: Rgb = (61, 163, 93)     # #3DA35D
WARN: Rgb = (201, 151, 26)    # #C9971A
BAD: Rgb = (208, 69, 60)      # #D0453C

LEG_LEFT: Rgb = (80, 200, 80)
LEG_LEFT_LIGHT: Rgb = (140, 220, 140)
LEG_LEFT_DARK: Rgb = (40, 150, 40)

LEG_RIGHT: Rgb = (255, 140, 60)
LEG_RIGHT_LIGHT: Rgb = (255, 190, 140)
LEG_RIGHT_DARK: Rgb = (190, 95, 30)

# Sprint-phase tints (drive / acceleration / max velocity / deceleration),
# plus jump-phase aliases for the same span-drawing code paths.
PHASE_COLORS: dict[str, Rgb] = {
    "drive": (255, 120, 60),
    "acceleration": (255, 200, 60),
    "max velocity": (80, 220, 120),
    "deceleration": (150, 150, 220),
    "countermovement": (255, 200, 60),
    "drive up": (255, 120, 60),
    "flight": (80, 220, 120),
    "landing": (150, 150, 220),
}

# Jump event markers that aren't left/right-paired.
TAKEOFF: Rgb = (255, 80, 80)
LANDING_EVENT: Rgb = (80, 220, 255)
CM_BOTTOM: Rgb = (220, 100, 220)

CONF_COLORS: dict[str, Rgb] = {"High": GOOD, "Medium": WARN, "Low": BAD}
SEVERITY_COLORS: dict[str, Rgb] = {"good": GOOD, "minor": WARN, "major": BAD}
GRADE_COLORS: dict[str, Rgb] = {"Good": GOOD, "Fair": WARN, "Poor": BAD}


# --- Helpers -----------------------------------------------------------

def hexs(rgb: Rgb) -> str:
    """`(208, 69, 60)` -> `"#d0453c"`, for HTML/QSS strings."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def qcolor(rgb: Rgb, alpha: int = 255) -> QColor:
    return QColor(rgb[0], rgb[1], rgb[2], alpha)


def bgr(rgb: Rgb) -> Rgb:
    """RGB -> BGR, for OpenCV consumers (`core/pose/skeleton.py`)."""
    return (rgb[2], rgb[1], rgb[0])


def make_swatch(rgb: Rgb, size: int = 10) -> QFrame:
    """A small solid-color square — a real widget standing in for the
    "■ left / ■ right" glyph-as-legend pattern."""
    dot = QFrame()
    dot.setFixedSize(size, size)
    dot.setStyleSheet(f"background: {hexs(rgb)}; border-radius: 2px;")
    return dot


def make_chip(text: str, rgb: Rgb, filled: bool = True) -> QLabel:
    """A small colored pill label — the status indicator everywhere a ✓/⚠/●
    glyph used to stand in. Real widget, not a unicode character doing a
    widget's job, so it scales/aligns/themes like the rest of the app."""
    label = QLabel(text.upper())
    bg = hexs(rgb) if filled else f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 40)"
    fg = "#0a0a0c" if filled else hexs(rgb)
    label.setStyleSheet(
        f"background: {bg}; color: {fg}; border-radius: 8px; "
        f"padding: 1px 8px; font-size: 10px; font-weight: 700; border: none;")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


# --- App-wide styling ----------------------------------------------------

def _palette() -> QPalette:
    p = QPalette()
    text = qcolor(TEXT)
    disabled = qcolor(TEXT_DISABLED)
    accent = qcolor(ACCENT)
    p.setColor(QPalette.ColorRole.Window, qcolor(BACKGROUND))
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, qcolor(SURFACE))
    p.setColor(QPalette.ColorRole.AlternateBase, qcolor(SURFACE_RAISED))
    p.setColor(QPalette.ColorRole.ToolTipBase, qcolor(SURFACE_RAISED))
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, qcolor(SURFACE_RAISED))
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, qcolor(BAD))
    p.setColor(QPalette.ColorRole.Link, accent)
    p.setColor(QPalette.ColorRole.Highlight, accent)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(10, 10, 12))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return p


def _stylesheet() -> str:
    bg, surface, raised, hover = hexs(BACKGROUND), hexs(SURFACE), hexs(SURFACE_RAISED), hexs(SURFACE_HOVER)
    hair = f"rgba({HAIRLINE[0]}, {HAIRLINE[1]}, {HAIRLINE[2]}, 30)"
    text, muted = hexs(TEXT), hexs(TEXT_MUTED)
    accent = hexs(ACCENT)
    return f"""
    QMainWindow, QDialog {{ background: {bg}; }}
    QWidget {{ color: {text}; }}
    QToolTip {{ background: {raised}; color: {text}; border: 1px solid {hair}; padding: 4px 6px; }}

    QToolBar {{ background: {surface}; border: none; spacing: 6px; padding: 5px 8px; }}
    QToolBar::separator {{ background: {hair}; width: 1px; margin: 4px 6px; }}
    QToolButton {{ background: transparent; border: none; border-radius: 6px; padding: 5px 9px; }}
    QToolButton:hover {{ background: {hover}; }}
    QToolButton:pressed {{ background: {raised}; }}
    QToolButton:disabled {{ color: {muted}; }}

    QStatusBar {{ background: {surface}; border-top: 1px solid {hair}; }}

    QDockWidget {{ titlebar-close-icon: none; color: {text}; }}
    QDockWidget::title {{
        background: {surface}; padding: 6px 8px; border-bottom: 1px solid {hair};
    }}

    QPushButton {{
        background: {raised}; border: 1px solid {hair}; border-radius: 6px;
        padding: 5px 12px;
    }}
    QPushButton:hover {{ background: {hover}; }}
    QPushButton:pressed {{ background: {surface}; }}
    QPushButton:disabled {{ color: {muted}; }}
    QPushButton:flat {{ background: transparent; border: none; padding: 2px; }}
    QPushButton:flat:hover {{ background: {hover}; border-radius: 4px; }}

    QComboBox {{
        background: {raised}; border: 1px solid {hair}; border-radius: 6px;
        padding: 4px 8px;
    }}
    QComboBox:hover {{ background: {hover}; }}
    QComboBox QAbstractItemView {{
        background: {raised}; selection-background-color: {accent};
        border: 1px solid {hair}; outline: none;
    }}

    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background: {raised}; border: 1px solid {hair}; border-radius: 4px;
        padding: 3px 6px; selection-background-color: {accent};
    }}

    QTableWidget {{
        background: {surface}; alternate-background-color: {raised};
        gridline-color: {hair}; border: 1px solid {hair}; border-radius: 6px;
        selection-background-color: rgba({ACCENT[0]}, {ACCENT[1]}, {ACCENT[2]}, 70);
    }}
    QHeaderView::section {{
        background: {raised}; color: {muted}; padding: 4px 6px; border: none;
        border-bottom: 1px solid {hair}; border-right: 1px solid {hair};
    }}
    QTableCornerButton::section {{ background: {raised}; border: none; }}

    QTextBrowser {{ background: {surface}; border: 1px solid {hair}; border-radius: 6px; }}

    QTabWidget::pane {{ border: 1px solid {hair}; border-radius: 6px; top: -1px; }}
    QTabBar::tab {{
        background: {surface}; color: {muted}; padding: 6px 16px;
        border-top-left-radius: 6px; border-top-right-radius: 6px;
        border: 1px solid {hair}; border-bottom: none;
    }}
    QTabBar::tab:selected {{ background: {raised}; color: {text}; border-bottom: 2px solid {accent}; }}
    QTabBar::tab:hover {{ background: {hover}; }}

    QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {raised}; border-radius: 4px; min-height: 24px; }}
    QScrollBar::handle:vertical:hover {{ background: {hover}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {raised}; border-radius: 4px; min-width: 24px; }}
    QScrollBar::handle:horizontal:hover {{ background: {hover}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

    QProgressBar {{
        background: {surface}; border: 1px solid {hair}; border-radius: 5px;
        text-align: center; color: {text};
    }}
    QProgressBar::chunk {{ background: {accent}; border-radius: 4px; }}

    QFrame#card {{ background: {surface}; border: 1px solid {hair}; border-radius: 10px; }}
    QFrame#heroTile {{ background: {raised}; border-radius: 8px; }}
    """


def apply(app: QApplication) -> None:
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setPalette(_palette())
    app.setStyleSheet(_stylesheet())
