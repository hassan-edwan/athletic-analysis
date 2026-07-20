"""Export golden JSON fixtures from the Python analysis core for the Swift port.

Two fixture kinds:
- "module": hand-built SprintMetrics (like tests/test_coaching.py) -> expected
  findings + radar. Exercises the coaching/diagnostics/radar tables without
  event detection.
- "pipeline": synthetic gait keypoints -> the full chain (smooth -> angles ->
  velocities -> events -> metrics -> findings -> radar). Proves the numerical
  ports (Butterworth filtfilt, gradients, percentiles) end to end.

The Python core is the reference implementation; regenerate with
    python tools/export_fixtures.py
whenever core behavior changes, then re-run `swift test` in
ios/AthleticAnalysisCore.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from athletic_analysis.core.angles import compute_angles  # noqa: E402
from athletic_analysis.core.coaching import analyze_sprint_form  # noqa: E402
from athletic_analysis.core.diagnostics import diagnose  # noqa: E402
from athletic_analysis.core.events import detect_gait_events  # noqa: E402
from athletic_analysis.core.filtering import smooth_keypoints  # noqa: E402
from athletic_analysis.core.metrics.sprint import (SprintMetrics,  # noqa: E402
                                                   StepRecord,
                                                   compute_sprint_metrics)
from athletic_analysis.core.pose.skeleton import KP  # noqa: E402
from athletic_analysis.core.radar import compute_sprint_radar  # noqa: E402
from athletic_analysis.core.velocity import compute_velocities  # noqa: E402

sys.path.insert(0, str(REPO / "tests"))
from conftest import make_sequence, standing_pose  # noqa: E402

OUT_DIR = (REPO / "ios" / "AthleticAnalysisCore" / "Tests"
           / "AthleticAnalysisCoreTests" / "Fixtures")

FPS = 100.0


def _clean(obj):
    """Recursively convert numpy types and NaN -> JSON-safe values."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _step(frame: int, side: str, *, contact=0.11, step_time=0.24, knee=155.0,
          thigh=75.0, trunk=6.0) -> StepRecord:
    return StepRecord(side=side, strike_frame=frame, toeoff_frame=frame + 11,
                      contact_time_s=contact, flight_time_s=0.13,
                      step_time_s=step_time, step_length=0.9, step_speed=3.8,
                      knee_angle_at_strike=knee, swing_thigh_angle=thigh,
                      trunk_lean_at_strike=trunk)


def _module_setup(trunk=6.0, knee=155.0, thigh=75.0, contact=0.11,
                  overstride_px=0.0, contact_by_side=None,
                  run_speed=None):
    T = 300
    kpts = make_sequence(T)
    kpts[:, :, 0] += np.arange(T)[:, None] * 8.0
    metrics = SprintMetrics()
    for i in range(1, 9):
        frame = 25 * i
        side = "left" if i % 2 == 0 else "right"
        c = contact
        if contact_by_side:
            c = contact_by_side[side]
        metrics.steps.append(_step(frame, side, trunk=trunk, knee=knee,
                                   thigh=thigh, contact=c))
        if overstride_px:
            sk = "l" if side == "left" else "r"
            kpts[frame, KP[f"{sk}_ankle"], 0] += overstride_px
    if run_speed is None:
        run_speed = np.full(T, 800.0)
    return kpts, metrics, {"run_speed": run_speed}


def _expected_findings(findings):
    out = []
    for f in findings:
        diag = diagnose(f)
        out.append({
            "key": f.key, "phase": f.phase, "severity": f.severity,
            "deviation": f.deviation, "value": f.value,
            "diagnosis_title": diag.title if diag else None,
        })
    return out


def _expected_radar(radar):
    if radar is None:
        return None
    return {
        "level": radar.level,
        "overall": radar.overall,
        "axes": [{"name": a.name, "score": a.score, "n_steps": a.n_steps}
                 for a in radar.axes],
    }


def _module_fixture(name, level="trained", **kwargs):
    kpts, metrics, velocities = _module_setup(**kwargs)
    findings = analyze_sprint_form(kpts, metrics, velocities, FPS, level)
    radar = compute_sprint_radar(kpts, metrics, velocities, FPS, level)
    return {
        "name": name,
        "kind": "module",
        "fps": FPS,
        "level": level,
        "keypoints": kpts,
        "run_speed": velocities["run_speed"],
        "steps": [asdict(s) for s in metrics.steps],
        "expected": {
            "findings": _expected_findings(findings),
            "radar": _expected_radar(radar),
        },
    }


def _gait_sequence(T=400, fps=FPS):
    """Synthetic alternating-gait clip that actually triggers event detection.

    Not biomechanically real — it only needs to drive both implementations
    identically: feet alternate ground dwell / sinusoidal swing while the body
    translates and the hips bounce slightly.
    """
    kpts = make_sequence(T)
    rng_x = np.arange(T) * 6.0
    kpts[:, :, 0] += rng_x[:, None]

    period = 50  # frames per full step cycle (stance 25 + swing 25)
    ground = {"l": 598.0, "r": 598.0}
    for t in range(T):
        for side, phase_off in (("l", 0), ("r", period // 2)):
            ph = (t + phase_off) % period
            if ph < 25:  # stance: foot planted at ground level
                lift = 0.0
            else:  # swing: raised, sinusoidal
                s = (ph - 25) / 25.0
                lift = 60.0 * math.sin(math.pi * s)
            for jname, base_y in ((f"{side}_big_toe", 600.0),
                                  (f"{side}_small_toe", 600.0),
                                  (f"{side}_heel", 598.0),
                                  (f"{side}_ankle", 580.0)):
                kpts[t, KP[jname], 1] = base_y - lift
        # Slight hip bounce, twice per cycle.
        bounce = 6.0 * math.sin(4 * math.pi * t / period)
        for jname in ("hip_center", "l_hip", "r_hip"):
            kpts[t, KP[jname], 1] += bounce
    _ = ground
    return kpts


def _pipeline_fixture(name, level="trained"):
    raw = _gait_sequence()
    smoothed = smooth_keypoints(raw, FPS)
    angles = compute_angles(smoothed)
    velocities, unit = compute_velocities(smoothed, FPS, None)
    events = detect_gait_events(smoothed, FPS)
    metrics = compute_sprint_metrics(smoothed, angles, events, FPS, None)
    findings = analyze_sprint_form(smoothed, metrics, velocities, FPS, level)
    radar = compute_sprint_radar(smoothed, metrics, velocities, FPS, level)
    return {
        "name": name,
        "kind": "pipeline",
        "fps": FPS,
        "level": level,
        "keypoints": raw,
        "expected": {
            "angles": {k: v for k, v in angles.items()},
            "run_speed": velocities["run_speed"],
            "velocity_unit": unit,
            "events": [{"frame": e.frame, "side": e.side, "kind": e.kind}
                       for e in events],
            "step_records": [asdict(s) for s in metrics.steps],
            "summary": {
                "cadence_spm": metrics.cadence_spm,
                "mean_contact_s": metrics.mean_contact_s,
                "mean_flight_s": metrics.mean_flight_s,
                "mean_step_length": metrics.mean_step_length,
                "mean_speed": metrics.mean_speed,
                "max_speed": metrics.max_speed,
                "mean_trunk_lean_deg": metrics.mean_trunk_lean_deg,
                "length_unit": metrics.length_unit,
                "body_height_px": metrics.body_height_px,
            },
            "findings": _expected_findings(findings),
            "radar": _expected_radar(radar),
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = [
        _module_fixture("module_good_form"),
        _module_fixture("module_trunk_fault", trunk=30.0),
        _module_fixture("module_long_contact", contact=0.30),
        _module_fixture("module_overstride", overstride_px=160.0),
        _module_fixture("module_contact_elite", level="elite", contact=0.13),
        _module_fixture("module_contact_dev", level="developmental",
                        contact=0.13),
        _module_fixture("module_asymmetric",
                        contact_by_side={"left": 0.14, "right": 0.09}),
        _module_fixture("module_drive_only", trunk=40.0, contact=0.18,
                        run_speed=np.linspace(100, 1000, 300)),
        _pipeline_fixture("pipeline_gait"),
    ]
    for fx in fixtures:
        path = OUT_DIR / f"{fx['name']}.json"
        path.write_text(json.dumps(_clean(fx)), encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")
    # Sanity summary for the pipeline fixture.
    pf = fixtures[-1]["expected"]
    print(f"pipeline events: {len(pf['events'])}, "
          f"steps: {len(pf['step_records'])}, "
          f"findings: {len(pf['findings'])}")


if __name__ == "__main__":
    main()
