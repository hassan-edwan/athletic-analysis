"""Pixel-to-meter calibration from a user-drawn line of known real length."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Calibration:
    meters_per_pixel: float

    @classmethod
    def from_line(cls, p1: tuple[float, float], p2: tuple[float, float],
                  real_length_m: float) -> "Calibration":
        px = math.dist(p1, p2)
        if px <= 0 or real_length_m <= 0:
            raise ValueError("Calibration line and real length must be non-zero")
        return cls(meters_per_pixel=real_length_m / px)

    def to_meters(self, pixels: float) -> float:
        return pixels * self.meters_per_pixel
