"""Playback controls: play/pause, frame stepping, slow-motion speed."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                               QWidget)

SPEEDS = [0.1, 0.25, 0.5, 1.0]


class Transport(QWidget):
    play_toggled = Signal(bool)  # True = play
    step = Signal(int)  # -1 / +1
    speed_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._playing = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self._prev = QPushButton("|<")
        self._play = QPushButton("Play")
        self._next = QPushButton(">|")
        for b in (self._prev, self._play, self._next):
            b.setFixedWidth(60)
            layout.addWidget(b)

        self._speed = QComboBox()
        for s in SPEEDS:
            self._speed.addItem(f"{s:g}x", s)
        self._speed.setCurrentIndex(len(SPEEDS) - 1)
        layout.addWidget(QLabel("Speed:"))
        layout.addWidget(self._speed)

        self._label = QLabel("frame - / -")
        layout.addStretch(1)
        layout.addWidget(self._label)

        self._prev.clicked.connect(lambda: self.step.emit(-1))
        self._next.clicked.connect(lambda: self.step.emit(+1))
        self._play.clicked.connect(self._toggle)
        self._speed.currentIndexChanged.connect(
            lambda: self.speed_changed.emit(self._speed.currentData()))

    def _toggle(self) -> None:
        self.set_playing(not self._playing)
        self.play_toggled.emit(self._playing)

    def set_playing(self, playing: bool) -> None:
        self._playing = playing
        self._play.setText("Pause" if playing else "Play")

    def speed(self) -> float:
        return self._speed.currentData()

    def set_position(self, frame: int, total: int, time_s: float) -> None:
        self._label.setText(f"frame {frame + 1} / {total}   t = {time_s:.3f} s")
