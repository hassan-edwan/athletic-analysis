"""Rules-based form analysis: grade measured mechanics against reference
ranges from sprint/jump biomechanics literature, tiered by athlete level.

Sprint steps are first assigned to a phase (drive / acceleration / max
velocity) from the speed profile, because "optimal" differs by phase: a 45°
trunk lean is textbook in the drive phase and a fault at max velocity. Ranges
also differ by athlete level — an elite sprinter's ~90 ms max-velocity ground
contact is not a fair target for a developmental athlete — so every check is
built for a chosen level and carries its literature source.

Sources (short-form; see README for full citations):
- Ground contact time: world-class 100 m ~0.089–0.095 s at max velocity
  (Sides, Salford thesis 2018; Nagahara et al.). Scaled up for trained /
  developmental and for earlier, longer-contact phases.
- Trunk: ~45° forward lean at block exit, ~vertical (0–10°) at max velocity
  (auptimo sprint-start review; World Athletics NSA).
- Front-side mechanics / knee lift and touchdown knee stiffness rise with
  level (elite land with a stiffer, more extended leg closer under the hips).
- Jump: countermovement depth, triple-extension at takeoff, and cushioned
  landing knee flexion from strength-and-conditioning norms.

Caveats: everything is 2D sagittal-view; phases are inferred *within the clip*
(a start-only clip has no true max-velocity data); ranges are coaching
reference points, not medical thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from athletic_analysis.core.angles import estimate_body_height_px, travel_direction
from athletic_analysis.core.confidence import (MetricConfidence,
                                               detection_factor,
                                               metric_confidence)
from athletic_analysis.core.metrics.jump import JumpMetrics
from athletic_analysis.core.metrics.sprint import SprintMetrics
from athletic_analysis.core.pose.skeleton import KP

SEVERITY_ORDER = {"major": 0, "minor": 1, "good": 2}

ATHLETE_LEVELS = ("developmental", "trained", "elite")


@dataclass
class FormFinding:
    phase: str
    metric: str
    value: float
    value_text: str
    target_text: str
    severity: str  # "good" | "minor" | "major"
    cue: str
    frame: int  # representative frame to jump to
    source: str = ""
    confidence: MetricConfidence | None = None
    key: str = ""  # machine metric key ("contact_ms", "trunk", …) or JumpMetrics attr
    deviation: str = ""  # "low" | "high" when out of band; "" when good


@dataclass
class _Check:
    metric: str
    unit: str  # for formatting: "deg" | "ms" | "spm" | "BH" | "m" | custom
    lo: float
    hi: float
    tol: float  # beyond range by more than this => "major"
    cue_low: str
    cue_high: str
    source: str = ""
    good_note: str = "In the optimal range."


def _fmt_value(value: float, unit: str) -> str:
    if not np.isfinite(value):
        return "–"
    if unit == "ms":
        return f"{value * 1000:.0f} ms"
    if unit == "deg":
        return f"{value:.0f}°"
    if unit == "spm":
        return f"{value:.0f} steps/min"
    return f"{value:.2f} {unit}"


def _fmt_range(check: _Check) -> str:
    if check.unit == "ms":
        return f"{check.lo * 1000:.0f}–{check.hi * 1000:.0f} ms"
    if check.unit == "deg":
        return f"{check.lo:.0f}–{check.hi:.0f}°"
    if check.unit == "spm":
        return f"{check.lo:.0f}–{check.hi:.0f} steps/min"
    return f"{check.lo:.2f}–{check.hi:.2f} {check.unit}"


def _evaluate(check: _Check, value: float, phase: str, frame: int,
              confidence: MetricConfidence | None = None,
              key: str = "") -> FormFinding | None:
    if value is None or not np.isfinite(value):
        return None
    if check.lo <= value <= check.hi:
        severity, cue, deviation = "good", check.good_note, ""
    else:
        dist = (check.lo - value) if value < check.lo else (value - check.hi)
        severity = "minor" if dist <= check.tol else "major"
        deviation = "low" if value < check.lo else "high"
        cue = check.cue_low if deviation == "low" else check.cue_high
    return FormFinding(phase=phase, metric=check.metric, value=value,
                       value_text=_fmt_value(value, check.unit),
                       target_text=_fmt_range(check), severity=severity,
                       cue=cue, frame=frame, source=check.source,
                       confidence=confidence, key=key, deviation=deviation)


# --- level-tiered reference numbers -------------------------------------------
# Each maps level -> (lo, hi, tol). Metrics not listed here are level-invariant
# technique targets defined inline in sprint_checks().

_CONTACT_S = {  # ground contact time, seconds, per phase
    "drive": {"developmental": (0.16, 0.26, 0.06), "trained": (0.13, 0.22, 0.06),
              "elite": (0.11, 0.18, 0.05)},
    "acceleration": {"developmental": (0.14, 0.22, 0.05),
                     "trained": (0.11, 0.18, 0.05), "elite": (0.10, 0.15, 0.04)},
    "max velocity": {"developmental": (0.11, 0.18, 0.05),
                     "trained": (0.095, 0.15, 0.04), "elite": (0.085, 0.12, 0.03)},
}
_CADENCE = {"developmental": (200, 300, 40), "trained": (220, 320, 35),
            "elite": (250, 340, 30)}
_THIGH = {"developmental": (45, 95, 15), "trained": (55, 100, 15),
          "elite": (65, 105, 12)}
_KNEE_TD = {  # knee angle at touchdown, deg, per phase
    "drive": {"developmental": (120, 165, 12), "trained": (125, 165, 12),
              "elite": (128, 165, 10)},
    "acceleration": {"developmental": (128, 168, 12),
                     "trained": (130, 168, 12), "elite": (135, 168, 10)},
    "max velocity": {"developmental": (135, 165, 12),
                     "trained": (140, 168, 10), "elite": (145, 170, 10)},
}
_CONTACT_SRC = "GCT: elite ~0.09 s at Vmax (Sides 2018; Nagahara)"
_TRUNK_SRC = "trunk ~45° block exit → vertical at Vmax (auptimo; World Athletics)"
_CAD_SRC = "cadence rises with level; ~4–5 Hz elite (Nagahara)"
_FS_SRC = "front-side mechanics / knee lift (Mann sprint model)"
_KNEE_SRC = "stiffer, more extended touchdown leg with level (Mann)"


def sprint_checks(level: str = "trained") -> dict[str, list[tuple[str, _Check]]]:
    """Phase -> [(metric_key, _Check)] for the given athlete level."""
    if level not in ATHLETE_LEVELS:
        level = "trained"

    def contact(phase: str) -> _Check:
        lo, hi, tol = _CONTACT_S[phase][level]
        return _Check("Ground contact time", "ms", lo, hi, tol,
                      cue_low="Contacts unusually short — verify capture FPS is "
                              "set correctly and that you're finishing each push.",
                      cue_high="Long ground contacts — build stiffness and faster "
                               "force production (bounds, sled/wall drills).",
                      source=_CONTACT_SRC)

    def knee_td(phase: str) -> _Check:
        lo, hi, tol = _KNEE_TD[phase][level]
        return _Check("Knee angle at touchdown", "deg", lo, hi, tol,
                      cue_low="Knee collapsing at touchdown — cue a stiffer, "
                              "spring-like leg on contact; build eccentric strength.",
                      cue_high="Landing on a near-locked leg — a slight knee bend "
                               "absorbs and returns energy; land closer to the hips.",
                      source=_KNEE_SRC)

    trunk_drive = _Check("Trunk lean", "deg", 20, 50, 12,
        cue_low="Too upright for the drive phase — stay low out of the start and "
                "push the ground back; let the body rise gradually.",
        cue_high="Leaning past ~50° — likely overreaching and losing balance.",
        source=_TRUNK_SRC)
    trunk_accel = _Check("Trunk lean", "deg", 8, 35, 10,
        cue_low="Popping up too early — hold a gradual rise through the transition.",
        cue_high="Still fully crouched — allow the trunk to rise as speed builds.",
        source=_TRUNK_SRC)
    trunk_maxv = _Check("Trunk lean", "deg", -2, 10, 8,
        cue_low="Leaning backward at top speed — run tall, eyes level.",
        cue_high="Excessive forward lean at max velocity kills front-side "
                 "mechanics — run tall with hips under the shoulders.",
        source=_TRUNK_SRC)

    def overstride(lo: float, hi: float) -> _Check:
        return _Check("Touchdown distance ahead of hip", "BH", lo, hi, 0.06,
            cue_low="Foot landing well behind the hips — check tracking.",
            cue_high="Overstriding — the foot planting ahead of the hips brakes "
                     "you; cue 'step down and back', not reaching out.",
            source=_FS_SRC)

    lo, hi, tol = _THIGH[level]
    thigh = _Check("Front-side knee lift (swing thigh)", "deg", lo, hi, tol,
        cue_low="Low knee lift — poor front-side mechanics; cue 'knees up, step "
                "over the opposite knee'.",
        cue_high="Thigh beyond expected range — check tracking.", source=_FS_SRC)
    clo, chi, ctol = _CADENCE[level]
    cadence = _Check("Cadence", "spm", clo, chi, ctol,
        cue_low="Low step rate — quicker, punchier steps; avoid overstriding.",
        cue_high="Very high cadence — possibly cutting strides short; finish each "
                 "push.", source=_CAD_SRC)

    return {
        "drive": [("trunk", trunk_drive), ("contact_ms", contact("drive")),
                  ("knee_strike", knee_td("drive")),
                  ("overstride", overstride(-0.12, 0.08))],
        "acceleration": [("trunk", trunk_accel),
                         ("contact_ms", contact("acceleration")),
                         ("knee_strike", knee_td("acceleration")),
                         ("overstride", overstride(-0.08, 0.12))],
        "max velocity": [("trunk", trunk_maxv),
                         ("contact_ms", contact("max velocity")),
                         ("cadence", cadence),
                         ("knee_strike", knee_td("max velocity")),
                         ("thigh", thigh), ("overstride", overstride(-0.02, 0.14))],
    }


def jump_checks(level: str = "trained", length_unit: str = "BH"
                ) -> list[tuple[str, _Check]]:
    """(metric_attr, _Check) list for jump grading. metric_attr is the
    JumpMetrics field the value comes from."""
    if level not in ATHLETE_LEVELS:
        level = "trained"
    # Countermovement depth: deeper acceptable band for stronger athletes.
    depth_by_level = {
        "m": {"developmental": (0.15, 0.40, 0.10), "trained": (0.20, 0.45, 0.10),
              "elite": (0.25, 0.50, 0.10)},
        "BH": {"developmental": (0.08, 0.24, 0.06), "trained": (0.10, 0.26, 0.06),
               "elite": (0.12, 0.30, 0.06)},
    }
    dlo, dhi, dtol = depth_by_level["m" if length_unit == "m" else "BH"][level]
    depth = _Check("Countermovement depth", length_unit, dlo, dhi, dtol,
        cue_low="Very shallow countermovement — you're not loading the legs; sink "
                "deeper and faster before takeoff.",
        cue_high="Very deep/slow countermovement — try a quicker, slightly "
                 "shallower dip to reverse faster.",
        source="CMJ depth norms (S&C literature)")
    knee_ext = _Check("Knee extension at takeoff", "deg", 160, 181, 10,
        cue_low="Leaving the ground with bent knees — finish the triple extension "
                "(hips, knees, ankles) completely.",
        cue_high="—", source="full triple extension at takeoff")
    trunk = _Check("Trunk lean at takeoff", "deg", -8, 25, 10,
        cue_low="Leaning back at takeoff — drive up, not backward.",
        cue_high="Excessive forward fold at takeoff bleeds vertical force — keep "
                 "the chest taller.", source="upright takeoff posture")
    landing = _Check("Peak knee flexion on landing", "deg", 60, 120, 15,
        cue_low="Collapsing very deep on landing — build eccentric strength to "
                "absorb in a stronger range.",
        cue_high="Stiff landing (knees barely bend) — absorb by sinking into hips "
                 "and knees; land 'quiet'.", source="cushioned landing (ACL-safe)")
    valgus = _Check("Knee/ankle separation ratio (frontal)", "x", 0.80, 1.80, 0.15,
        cue_low="Knees caving inward on landing (valgus) — cue 'knees over toes'; "
                "strengthen glutes/abductors.",
        cue_high="Knees pushed far outside the ankles — uncommon; check tracking.",
        source="frontal-plane valgus proxy")
    return [
        ("countermovement_depth", depth),
        ("knee_angle_at_takeoff", knee_ext),
        ("trunk_lean_at_takeoff", trunk),
        ("peak_knee_flexion_landing", landing),
        ("knee_ankle_sep_ratio_landing", valgus),
    ]


# --- plain-language "what is this measuring" -----------------------------
# Independent of whether a value is good or bad (that's what Check.cue_low/
# cue_high are for) — this is the neutral sentence a metric name alone never
# carries. One entry per sprint_checks() key and per jump_checks() attr;
# reused everywhere a metric name appears on its own (column headers, chart
# captions, checkbox tooltips) so a user never has to already know the
# vocabulary to read the app.
_METRIC_HELP: dict[str, str] = {
    "trunk": "How far forward your torso tips at that instant. Different "
             "phases call for different amounts — a big forward lean out of "
             "the start, close to upright at top speed.",
    "contact_ms": "How long your foot stays on the ground each step. "
                  "Shorter generally means a stiffer, faster step.",
    "knee_strike": "How bent your leg is the instant your foot lands. Too "
                   "bent and the leg absorbs instead of springing back; too "
                   "straight and there's no give.",
    "thigh": "How high your swing leg's thigh rises relative to vertical. "
             "More lift usually means better front-side mechanics at speed.",
    "cadence": "How many steps you take per minute. Higher generally means "
               "quicker turnover, but only if each step still finishes its "
               "push.",
    "overstride": "How far in front of your hips your foot lands. Landing "
                  "too far ahead brakes you with every step.",
    "countermovement_depth": "How far your hips dip down before you jump. "
                             "Deeper can mean more force, but too deep or "
                             "slow costs you the stretch-reflex advantage.",
    "knee_angle_at_takeoff": "How straight your legs are the instant you "
                             "leave the ground. Full extension means you're "
                             "not leaving power on the table.",
    "trunk_lean_at_takeoff": "How upright your torso is as you leave the "
                             "ground. Leaning too far forward or back bleeds "
                             "vertical force sideways.",
    "peak_knee_flexion_landing": "How much your knees bend to absorb the "
                                 "landing. Too stiff transmits impact up the "
                                 "joint chain; too deep can mean poor "
                                 "control.",
    "knee_ankle_sep_ratio_landing": "A frontal-view check for knees caving "
                                    "inward (valgus) on landing.",
}


def metric_help(key: str) -> str:
    """One plain-language sentence describing what `key` measures — empty
    string for an unknown key rather than raising, since this is always used
    for optional supplementary UI text (tooltips, captions)."""
    return _METRIC_HELP.get(key, "")


def segment_phases(run_speed: np.ndarray | None, fps: float) -> list[tuple[int, int, str]]:
    """Per-frame phase segmentation of a sprint for display: contiguous
    (start_frame, end_frame, phase) spans. Before the speed peak frames are
    drive / acceleration / max velocity by speed ratio; after the peak,
    anything clearly below max is labeled deceleration."""
    if run_speed is None:
        return []
    rs = np.asarray(run_speed, dtype=np.float64)
    finite = np.isfinite(rs)
    if finite.sum() < 5:
        return []
    vmax = float(np.nanmax(rs))
    if not np.isfinite(vmax) or vmax <= 0:
        return []
    peak = int(np.nanargmax(rs))
    ratio = rs / vmax

    labels: list[str | None] = []
    prev: str | None = None
    for i, r in enumerate(ratio):
        if not np.isfinite(r):
            labels.append(prev)
            continue
        if r >= 0.93:
            cur = "max velocity"
        elif i <= peak:
            cur = "drive" if r < 0.70 else "acceleration"
        else:
            cur = "deceleration"
        labels.append(cur)
        prev = cur
    first = next((c for c in labels if c), None)
    if first is None:
        return []
    for i, c in enumerate(labels):
        if c is not None:
            break
        labels[i] = first

    spans: list[list] = []
    for i, c in enumerate(labels):
        if spans and spans[-1][2] == c:
            spans[-1][1] = i
        else:
            spans.append([i, i, c])
    # Absorb blips shorter than ~0.15 s into the preceding span.
    min_len = max(3, round(0.15 * fps))
    merged: list[list] = []
    for sp in spans:
        if merged and sp[1] - sp[0] + 1 < min_len:
            merged[-1][1] = sp[1]
        elif merged and merged[-1][2] == sp[2]:
            merged[-1][1] = sp[1]
        else:
            merged.append(sp)
    return [(int(a), int(b), c) for a, b, c in merged]


def plot_target_bands(level: str = "trained"
                      ) -> dict[str, dict[str, tuple[float, float]]]:
    """Optimal ranges valid across a whole phase (drawable as graph bands):
    plot-series key -> {phase: (lo, hi)}. Touchdown-instant metrics (knee angle
    at strike, contact time) are excluded — a full-curve band would mislead."""
    out: dict[str, dict[str, tuple[float, float]]] = {
        "trunk_lean": {}, "thigh_l": {}, "thigh_r": {}}
    for phase, checks in sprint_checks(level).items():
        for key, check in checks:
            if key == "trunk":
                out["trunk_lean"][phase] = (check.lo, check.hi)
            elif key == "thigh":
                out["thigh_l"][phase] = (check.lo, check.hi)
                out["thigh_r"][phase] = (check.lo, check.hi)
    return out


def _phase_of_step(strike_frame: int, run_speed: np.ndarray | None,
                   trunk_lean: float) -> str:
    ratio = np.nan
    if run_speed is not None and 0 <= strike_frame < len(run_speed):
        vmax = np.nanmax(run_speed) if np.isfinite(run_speed).any() else np.nan
        if np.isfinite(vmax) and vmax > 0 and np.isfinite(run_speed[strike_frame]):
            ratio = run_speed[strike_frame] / vmax
    if np.isfinite(ratio):
        if ratio >= 0.93:
            return "max velocity"
        if ratio < 0.70:
            return "drive"
        return "acceleration"
    # No usable speed: fall back to posture.
    if np.isfinite(trunk_lean) and trunk_lean > 25:
        return "drive"
    return "max velocity"


def _overstride(kpts: np.ndarray, strike_frame: int, side: str,
                direction: float, body_h_px: float) -> float:
    """Horizontal distance (body-heights) the striking ankle lands ahead of the
    hip; positive = ahead of the body (braking)."""
    if not np.isfinite(body_h_px) or body_h_px <= 0:
        return float("nan")
    ankle = kpts[strike_frame, KP[f"{side[0]}_ankle"]]
    hip = kpts[strike_frame, KP["hip_center"]]
    if ankle[2] < 0.3 or hip[2] < 0.3:
        return float("nan")
    return float((ankle[0] - hip[0]) * direction / body_h_px)


@dataclass
class PhaseBucket:
    """Per-phase step measurements, shared by form grading and radar scoring."""
    values: dict[str, list[float]] = field(default_factory=dict)  # metric key -> per-step
    strike_frames: list[int] = field(default_factory=list)
    contact_frames: list[float] = field(default_factory=list)  # contact_time_s * fps
    step_frames: list[float] = field(default_factory=list)  # step_time_s * fps


def median_of(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.median(finite)) if len(finite) else float("nan")


def bucket_sprint_steps(kpts: np.ndarray, sprint: SprintMetrics,
                        velocities: dict[str, np.ndarray],
                        fps: float) -> dict[str, PhaseBucket]:
    """Assign each step to drive / acceleration / max velocity and collect
    per-metric value lists keyed like sprint_checks()."""
    run_speed = velocities.get("run_speed")
    direction = travel_direction(kpts)
    body_h = estimate_body_height_px(kpts)
    buckets: dict[str, PhaseBucket] = {}
    for step in sprint.steps:
        phase = _phase_of_step(step.strike_frame, run_speed,
                               step.trunk_lean_at_strike)
        b = buckets.setdefault(phase, PhaseBucket())
        b.strike_frames.append(step.strike_frame)
        v = b.values
        v.setdefault("trunk", []).append(step.trunk_lean_at_strike)
        v.setdefault("contact_ms", []).append(step.contact_time_s)
        v.setdefault("knee_strike", []).append(step.knee_angle_at_strike)
        v.setdefault("thigh", []).append(step.swing_thigh_angle)
        v.setdefault("cadence", []).append(
            60.0 / step.step_time_s if np.isfinite(step.step_time_s)
            and step.step_time_s > 0 else float("nan"))
        v.setdefault("overstride", []).append(
            _overstride(kpts, step.strike_frame, step.side, direction, body_h))
        if np.isfinite(step.contact_time_s):
            b.contact_frames.append(step.contact_time_s * fps)
        if np.isfinite(step.step_time_s):
            b.step_frames.append(step.step_time_s * fps)
    return buckets


# Joints whose tracking quality bounds each metric's confidence.
_METRIC_JOINTS = {
    "trunk": ["l_shoulder", "r_shoulder", "l_hip", "r_hip"],
    "contact_ms": ["l_heel", "r_heel", "l_big_toe", "r_big_toe"],
    "knee_strike": ["l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"],
    "thigh": ["l_hip", "r_hip", "l_knee", "r_knee"],
    "overstride": ["hip_center", "l_ankle", "r_ankle"],
    "cadence": ["hip_center"],
}


def analyze_sprint_form(kpts: np.ndarray, sprint: SprintMetrics | None,
                        velocities: dict[str, np.ndarray], fps: float,
                        level: str = "trained") -> list[FormFinding]:
    if sprint is None or not sprint.steps:
        return []
    checks_by_phase = sprint_checks(level)
    buckets = bucket_sprint_steps(kpts, sprint, velocities, fps)

    def conf_for(key: str, bucket: PhaseBucket) -> MetricConfidence:
        strikes = bucket.strike_frames
        detection = None
        if key in _METRIC_JOINTS and kpts is not None:
            detection = detection_factor(kpts, _METRIC_JOINTS[key], strikes)
        spanned = None
        if key == "contact_ms":
            cf = bucket.contact_frames
            spanned = float(np.median(cf)) if cf else 0.0
        elif key == "cadence":
            sf = bucket.step_frames
            spanned = float(np.median(sf)) if sf else 0.0
        return metric_confidence(detection=detection, frames_spanned=spanned,
                                 n_samples=len(strikes))

    findings: list[FormFinding] = []
    for phase in ("drive", "acceleration", "max velocity"):
        if phase not in buckets:
            continue
        bucket = buckets[phase]
        rep_frame = bucket.strike_frames[0]
        for key, check in checks_by_phase[phase]:
            finding = _evaluate(check, median_of(bucket.values.get(key, [])),
                                phase, rep_frame, conf_for(key, bucket), key=key)
            if finding is not None:
                findings.append(finding)
    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.frame))
    return findings


_JUMP_METRIC_JOINTS = {
    "countermovement_depth": ["hip_center"],
    "knee_angle_at_takeoff": ["l_hip", "r_hip", "l_knee", "r_knee", "l_ankle",
                              "r_ankle"],
    "trunk_lean_at_takeoff": ["l_shoulder", "r_shoulder", "l_hip", "r_hip"],
    "peak_knee_flexion_landing": ["l_hip", "r_hip", "l_knee", "r_knee",
                                  "l_ankle", "r_ankle"],
    "knee_ankle_sep_ratio_landing": ["l_knee", "r_knee", "l_ankle", "r_ankle"],
}


def analyze_jump_form(jump: JumpMetrics | None, kpts: np.ndarray | None = None,
                      fps: float = 30.0, level: str = "trained"
                      ) -> list[FormFinding]:
    if jump is None or jump.takeoff_frame < 0:
        return []
    findings: list[FormFinding] = []
    for attr, check in jump_checks(level, jump.length_unit):
        value = getattr(jump, attr, float("nan"))
        frame = (jump.landing_frame if "landing" in attr else jump.takeoff_frame)
        detection = None
        if kpts is not None and attr in _JUMP_METRIC_JOINTS:
            detection = detection_factor(kpts, _JUMP_METRIC_JOINTS[attr], [frame])
        uncal = (check.unit in ("BH",) and attr == "countermovement_depth")
        conf = metric_confidence(detection=detection, uncalibrated_distance=uncal)
        finding = _evaluate(check, value, "jump", frame, conf, key=attr)
        if finding is not None:
            findings.append(finding)
    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.frame))
    return findings


def summarize(findings: list[FormFinding]) -> str:
    if not findings:
        return "No form checks available — run analysis first."
    good = sum(1 for f in findings if f.severity == "good")
    minor = sum(1 for f in findings if f.severity == "minor")
    major = sum(1 for f in findings if f.severity == "major")
    return (f"{good}/{len(findings)} checks in optimal range · "
            f"{minor} minor · {major} major")
