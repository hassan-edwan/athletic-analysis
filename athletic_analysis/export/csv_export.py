"""CSV exports: per-frame kinematics and per-event metrics."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from athletic_analysis.core.diagnostics import diagnose
from athletic_analysis.core.pose.skeleton import KP
from athletic_analysis.core.session import AnalysisSession


def export_frames_csv(session: AnalysisSession, path: str | Path) -> None:
    """One row per frame: time, joint angles, key landmark positions."""
    angle_keys = sorted(session.angles.keys())
    vel_keys = sorted(session.velocities.keys())
    vel_unit = session.velocity_unit.replace("/", "_per_")
    landmark_cols = ["hip_center", "l_ankle", "r_ankle", "l_big_toe", "r_big_toe",
                     "l_heel", "r_heel"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["frame", "time_s"] + angle_keys
        header += [f"{k}_{vel_unit}" for k in vel_keys]
        for name in landmark_cols:
            header += [f"{name}_x", f"{name}_y", f"{name}_conf"]
        writer.writerow(header)
        T = len(session.keypoints)
        for t in range(T):
            row: list[object] = [t, round(t / session.fps, 4)]
            row += [round(float(session.angles[k][t]), 2)
                    if np.isfinite(session.angles[k][t]) else ""
                    for k in angle_keys]
            row += [round(float(session.velocities[k][t]), 3)
                    if np.isfinite(session.velocities[k][t]) else ""
                    for k in vel_keys]
            for name in landmark_cols:
                x, y, c = session.keypoints[t, KP[name]]
                row += [round(float(x), 1), round(float(y), 1), round(float(c), 2)]
            writer.writerow(row)


def export_metrics_csv(session: AnalysisSession, path: str | Path) -> None:
    """Mode-dependent: per-step table (sprint) or key/value list (jump)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if session.mode == "sprint" and session.sprint_metrics:
            m = session.sprint_metrics
            writer.writerow(["# sprint summary"])
            writer.writerow(["cadence_spm", m.cadence_spm])
            writer.writerow(["mean_contact_s", m.mean_contact_s])
            writer.writerow(["mean_flight_s", m.mean_flight_s])
            writer.writerow([f"mean_step_length_{m.length_unit}", m.mean_step_length])
            writer.writerow([f"mean_speed_{m.length_unit}_per_s", m.mean_speed])
            writer.writerow([f"max_speed_{m.length_unit}_per_s", m.max_speed])
            writer.writerow(["mean_trunk_lean_deg", m.mean_trunk_lean_deg])
            writer.writerow([])
            if m.steps:
                keys = list(asdict(m.steps[0]).keys())
                writer.writerow(keys)
                for step in m.steps:
                    writer.writerow(asdict(step).values())
        elif session.mode == "jump" and session.jump_metrics:
            for key, value in asdict(session.jump_metrics).items():
                writer.writerow([key, value])
        findings = (session.sprint_form if session.mode == "sprint"
                    else session.jump_form)
        if findings:
            writer.writerow([])
            tnote = (", transforms: " + "+".join(session.transforms)
                     if session.transforms else "")
            rnote = f", rotation: {session.rotation}°" if session.rotation else ""
            writer.writerow([f"# form analysis (level: {session.athlete_level}, "
                             f"model: {session.model_tier}{rnote}{tnote})"])
            if session.quality is not None:
                writer.writerow(["# analysis quality", session.quality.level,
                                 "; ".join(session.quality.notes)])
            writer.writerow(["severity", "phase", "metric", "measured",
                             "optimal", "confidence", "limiter", "source", "cue",
                             "likely_causes", "likely_muscle_factors",
                             "corrective_drills"])
            for f in findings:
                conf = f.confidence.level if f.confidence else ""
                limiter = f.confidence.limiter if f.confidence else ""
                diag = diagnose(f)
                causes = "; ".join(diag.technical_causes) if diag else ""
                muscles = "; ".join(diag.muscle_factors) if diag else ""
                drills = "; ".join(diag.drills) if diag else ""
                writer.writerow([f.severity, f.phase, f.metric, f.value_text,
                                 f.target_text, conf, limiter, f.source, f.cue,
                                 causes, muscles, drills])
        if session.mode == "sprint" and session.sprint_radar is not None:
            writer.writerow([])
            writer.writerow([f"# sprint factor scores 0-100 "
                             f"(level: {session.sprint_radar.level})"])
            writer.writerow(["factor", "score", "steps", "detail"])
            for axis in session.sprint_radar.axes:
                score = (f"{axis.score:.1f}" if np.isfinite(axis.score) else "")
                writer.writerow([axis.name, score, axis.n_steps, axis.detail])
            overall = session.sprint_radar.overall
            writer.writerow(["overall",
                             f"{overall:.1f}" if np.isfinite(overall) else "",
                             "", ""])
