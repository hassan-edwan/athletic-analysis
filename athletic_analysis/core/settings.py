"""Tiny JSON-backed user settings (pose model tier, athlete level).

Kept dependency-free and defensive: any read/parse error falls back to
defaults so a corrupt file never blocks the app.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_TIERS = ("Fast", "Balanced", "Accurate")
ATHLETE_LEVELS = ("developmental", "trained", "elite")

# App-facing model tier -> rtmlib mode.
TIER_TO_MODE = {"Fast": "lightweight", "Balanced": "balanced",
                "Accurate": "performance"}


def _settings_path() -> Path:
    return Path.home() / ".athletic_analysis" / "settings.json"


@dataclass
class Settings:
    model_tier: str = "Balanced"
    athlete_level: str = "trained"

    def normalized(self) -> "Settings":
        if self.model_tier not in MODEL_TIERS:
            self.model_tier = "Balanced"
        if self.athlete_level not in ATHLETE_LEVELS:
            self.athlete_level = "trained"
        return self

    @classmethod
    def load(cls) -> "Settings":
        path = _settings_path()
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        return cls(
            model_tier=data.get("model_tier", "Balanced"),
            athlete_level=data.get("athlete_level", "trained"),
        ).normalized()

    def save(self) -> None:
        path = _settings_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except OSError:
            pass  # settings are a convenience, never fatal
