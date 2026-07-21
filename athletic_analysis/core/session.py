"""AnalysisSession: holds video + pose data, runs the analysis pipeline, and
persists raw results to a JSON sidecar next to the video.

Only the expensive/irreproducible inputs are serialized (raw keypoints, mode,
calibration); filtered trajectories, angles, events and metrics are recomputed
on load — they're cheap and this keeps the file format tiny and stable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from athletic_analysis.core.angles import compute_angles
from athletic_analysis.core.calibration import Calibration
from athletic_analysis.core.coaching import (FormFinding, analyze_jump_form,
                                             analyze_sprint_form)
from athletic_analysis.core.confidence import ClipQuality, clip_quality
from athletic_analysis.core.events import (GaitEvent, JumpPhases,
                                           detect_gait_events, detect_jump)
from athletic_analysis.core.filtering import smooth_keypoints
from athletic_analysis.core.metrics.jump import JumpMetrics, compute_jump_metrics
from athletic_analysis.core.metrics.sprint import SprintMetrics, compute_sprint_metrics
from athletic_analysis.core.quality import (TrackingQuality, ViewClassification,
                                            classify_view, tracking_quality)
from athletic_analysis.core.radar import SprintRadar, compute_sprint_radar
from athletic_analysis.core.velocity import compute_velocities

FORMAT_VERSION = 3


@dataclass
class AnalysisSession:
    video_path: str
    fps: float
    mode: str = "sprint"  # "sprint" | "jump"
    keypoints_raw: np.ndarray | None = None  # (T, 26, 3)
    calibration: Calibration | None = None
    athlete_level: str = "trained"  # developmental | trained | elite
    model_tier: str = "Balanced"  # Fast | Balanced | Accurate that produced this
    rotation: int = 0  # display rotation applied before analysis
    transforms: list[str] = field(default_factory=list)  # preprocessing applied

    # Derived (recomputed, never serialized):
    keypoints: np.ndarray | None = field(default=None, repr=False)
    angles: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    velocities: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    velocity_unit: str = field(default="BH/s", repr=False)
    gait_events: list[GaitEvent] = field(default_factory=list, repr=False)
    jump_phases: JumpPhases | None = field(default=None, repr=False)
    sprint_metrics: SprintMetrics | None = field(default=None, repr=False)
    jump_metrics: JumpMetrics | None = field(default=None, repr=False)
    sprint_form: list[FormFinding] = field(default_factory=list, repr=False)
    jump_form: list[FormFinding] = field(default_factory=list, repr=False)
    sprint_radar: SprintRadar | None = field(default=None, repr=False)
    quality: ClipQuality | None = field(default=None, repr=False)
    tracking: TrackingQuality | None = field(default=None, repr=False)
    view: ViewClassification | None = field(default=None, repr=False)

    @property
    def has_pose(self) -> bool:
        return self.keypoints_raw is not None and len(self.keypoints_raw) > 0

    def recompute(self) -> None:
        """Full pipeline after keypoints / mode / calibration change."""
        if not self.has_pose:
            return
        self.keypoints = smooth_keypoints(self.keypoints_raw, self.fps)
        self.angles = compute_angles(self.keypoints)
        self.tracking = tracking_quality(self.keypoints)
        self.view = classify_view(self.keypoints)
        self.velocities, self.velocity_unit = compute_velocities(
            self.keypoints, self.fps, self.calibration)
        self.gait_events = detect_gait_events(self.keypoints, self.fps)
        self.jump_phases = detect_jump(self.keypoints, self.fps)
        self.sprint_metrics = compute_sprint_metrics(
            self.keypoints, self.angles, self.gait_events, self.fps,
            self.calibration, subframe=True)
        self.jump_metrics = compute_jump_metrics(
            self.keypoints, self.angles, self.jump_phases, self.fps, self.calibration)
        self.sprint_form = analyze_sprint_form(
            self.keypoints, self.sprint_metrics, self.velocities, self.fps,
            self.athlete_level, plausibility=self.tracking.plausibility)
        self.jump_form = analyze_jump_form(
            self.jump_metrics, self.keypoints, self.fps, self.athlete_level,
            view=self.view.view)
        self.sprint_radar = compute_sprint_radar(
            self.keypoints, self.sprint_metrics, self.velocities, self.fps,
            self.athlete_level)
        self.quality = clip_quality(
            self.keypoints, self.fps, self.calibration is not None,
            mean_plausibility=self.tracking.mean_plausibility,
            view=self.view.view)

    # --- persistence -------------------------------------------------------

    @staticmethod
    def sidecar_path(video_path: str | Path) -> Path:
        return Path(str(video_path) + ".analysis.json")

    def save(self) -> Path:
        path = self.sidecar_path(self.video_path)
        data = {
            "version": FORMAT_VERSION,
            "fps": self.fps,
            "mode": self.mode,
            "athlete_level": self.athlete_level,
            "model_tier": self.model_tier,
            "rotation": self.rotation,
            "transforms": self.transforms,
            "meters_per_pixel": (self.calibration.meters_per_pixel
                                 if self.calibration else None),
            "keypoints_raw": (np.round(self.keypoints_raw, 2).tolist()
                              if self.has_pose else None),
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    @classmethod
    def load(cls, video_path: str | Path, fps: float) -> "AnalysisSession | None":
        path = cls.sidecar_path(video_path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        mpp = data.get("meters_per_pixel")
        raw = data.get("keypoints_raw")
        session = cls(
            video_path=str(video_path),
            fps=float(data.get("fps", fps)),
            mode=data.get("mode", "sprint"),
            keypoints_raw=np.asarray(raw, dtype=np.float32) if raw else None,
            calibration=Calibration(mpp) if mpp else None,
            athlete_level=data.get("athlete_level", "trained"),
            model_tier=data.get("model_tier", "Balanced"),
            rotation=int(data.get("rotation", 0)),
            transforms=list(data.get("transforms", [])),
        )
        session.recompute()
        return session
