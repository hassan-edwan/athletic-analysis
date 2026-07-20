import numpy as np

from athletic_analysis.core.coaching import (analyze_jump_form,
                                             analyze_sprint_form, jump_checks,
                                             metric_help, plot_target_bands,
                                             segment_phases, sprint_checks,
                                             summarize)
from athletic_analysis.core.metrics.jump import JumpMetrics
from athletic_analysis.core.metrics.sprint import SprintMetrics, StepRecord
from athletic_analysis.core.pose.skeleton import KP
from tests.conftest import make_sequence

FPS = 100.0


def _step(frame: int, side: str, *, contact=0.11, step_time=0.24, knee=155.0,
          thigh=75.0, trunk=6.0) -> StepRecord:
    return StepRecord(side=side, strike_frame=frame, toeoff_frame=frame + 11,
                      contact_time_s=contact, flight_time_s=0.13,
                      step_time_s=step_time, step_length=0.9, step_speed=3.8,
                      knee_angle_at_strike=knee, swing_thigh_angle=thigh,
                      trunk_lean_at_strike=trunk)


def _sprint_setup(trunk=6.0, knee=155.0, thigh=75.0, contact=0.11,
                  overstride_px=0.0):
    """Constant-speed sprint (=> max velocity phase) with adjustable faults."""
    T = 300
    kpts = make_sequence(T)
    kpts[:, :, 0] += np.arange(T)[:, None] * 8.0
    metrics = SprintMetrics()
    for i in range(1, 9):
        frame = 25 * i
        metrics.steps.append(_step(frame, "left" if i % 2 == 0 else "right",
                                   trunk=trunk, knee=knee, thigh=thigh,
                                   contact=contact))
        if overstride_px:
            side = "l" if i % 2 == 0 else "r"
            kpts[frame, KP[f"{side}_ankle"], 0] += overstride_px
    velocities = {"run_speed": np.full(T, 800.0)}  # constant => max velocity
    return kpts, metrics, velocities


def test_good_form_all_checks_pass():
    kpts, metrics, vel = _sprint_setup()
    findings = analyze_sprint_form(kpts, metrics, vel, FPS)
    assert findings
    assert all(f.phase == "max velocity" for f in findings)
    assert all(f.severity == "good" for f in findings), \
        [(f.metric, f.value_text, f.severity) for f in findings]


def test_excessive_trunk_lean_flagged():
    kpts, metrics, vel = _sprint_setup(trunk=30.0)
    findings = analyze_sprint_form(kpts, metrics, vel, FPS)
    trunk = next(f for f in findings if f.metric == "Trunk lean")
    assert trunk.severity == "major"
    assert "lean" in trunk.cue.lower()


def test_overstride_flagged():
    # Ankles shifted ~25% of body height ahead of the hip at each strike.
    kpts, metrics, vel = _sprint_setup(overstride_px=160.0)
    findings = analyze_sprint_form(kpts, metrics, vel, FPS)
    over = next(f for f in findings
                if f.metric == "Touchdown distance ahead of hip")
    assert over.severity in ("minor", "major")
    assert "overstrid" in over.cue.lower()


def test_drive_phase_allows_big_lean():
    kpts, metrics, vel = _sprint_setup(trunk=40.0, contact=0.16)
    # Rising speed profile puts the early steps in the drive phase.
    vel = {"run_speed": np.linspace(100, 1000, 300)}
    findings = analyze_sprint_form(kpts, metrics, vel, FPS)
    drive_trunk = [f for f in findings
                   if f.metric == "Trunk lean" and f.phase == "drive"]
    assert drive_trunk and drive_trunk[0].severity == "good"


def test_no_steps_no_findings():
    kpts = make_sequence(50)
    assert analyze_sprint_form(kpts, SprintMetrics(), {}, FPS) == []
    assert analyze_jump_form(None) == []


def _jump_metrics(**overrides) -> JumpMetrics:
    m = JumpMetrics(takeoff_frame=150, landing_frame=190, flight_time_s=0.4,
                    jump_height_flight_m=0.20, takeoff_velocity_m_s=1.96,
                    hip_rise=0.18, countermovement_depth=0.18, length_unit="BH",
                    knee_angle_at_takeoff=172.0, hip_angle_at_takeoff=170.0,
                    trunk_lean_at_takeoff=10.0, peak_knee_flexion_landing=95.0,
                    knee_ankle_sep_ratio_landing=1.1)
    for key, value in overrides.items():
        setattr(m, key, value)
    return m


def test_good_jump_all_checks_pass():
    findings = analyze_jump_form(_jump_metrics())
    assert findings
    assert all(f.severity == "good" for f in findings), \
        [(f.metric, f.value_text, f.severity) for f in findings]


def test_stiff_landing_flagged():
    findings = analyze_jump_form(_jump_metrics(peak_knee_flexion_landing=155.0))
    landing = next(f for f in findings
                   if f.metric == "Peak knee flexion on landing")
    assert landing.severity == "major"
    assert "stiff" in landing.cue.lower()


def test_knee_valgus_flagged():
    findings = analyze_jump_form(_jump_metrics(knee_ankle_sep_ratio_landing=0.55))
    valgus = next(f for f in findings if "separation" in f.metric.lower())
    assert valgus.severity == "major"
    assert "valgus" in valgus.cue.lower() or "caving" in valgus.cue.lower()


def test_summary_counts():
    findings = analyze_jump_form(_jump_metrics(peak_knee_flexion_landing=155.0))
    text = summarize(findings)
    assert "major" in text and "/" in text


def test_segment_phases_accelerating_run():
    speed = np.linspace(50, 1000, 400)  # steadily accelerating clip
    spans = segment_phases(speed, FPS)
    names = [n for _a, _b, n in spans]
    assert names == ["drive", "acceleration", "max velocity"]
    # Spans must tile the clip in order without gaps.
    assert spans[0][0] == 0 and spans[-1][1] == 399
    for (a, b, _n), (c, _d, _m) in zip(spans, spans[1:]):
        assert c == b + 1


def test_segment_phases_constant_speed_is_all_max_velocity():
    spans = segment_phases(np.full(200, 800.0), FPS)
    assert len(spans) == 1 and spans[0][2] == "max velocity"


def test_segment_phases_deceleration_after_peak():
    speed = np.concatenate([np.linspace(100, 1000, 200), np.linspace(1000, 500, 200)])
    spans = segment_phases(speed, FPS)
    assert spans[-1][2] == "deceleration"


def test_segment_phases_handles_missing_data():
    assert segment_phases(None, FPS) == []
    assert segment_phases(np.full(100, np.nan), FPS) == []


def test_plot_target_bands_phase_dependent():
    bands = plot_target_bands()
    assert bands["trunk_lean"]["drive"][0] > bands["trunk_lean"]["max velocity"][1], \
        "drive-phase trunk target must sit above the max-velocity target"
    assert "max velocity" in bands["thigh_l"]


# --- tiered ranges ------------------------------------------------------------

def test_elite_contact_tighter_than_developmental():
    dev = dict(sprint_checks("developmental")["max velocity"])
    elite = dict(sprint_checks("elite")["max velocity"])
    assert elite["contact_ms"].hi < dev["contact_ms"].hi
    assert elite["contact_ms"].lo < dev["contact_ms"].lo


def test_same_contact_graded_differently_by_level():
    # 130 ms max-velocity contact: fine for developmental, slow for elite.
    kpts, metrics, vel = _sprint_setup(contact=0.13)
    dev = analyze_sprint_form(kpts, metrics, vel, FPS, "developmental")
    elite = analyze_sprint_form(kpts, metrics, vel, FPS, "elite")
    dev_c = next(f for f in dev if f.metric == "Ground contact time")
    elite_c = next(f for f in elite if f.metric == "Ground contact time")
    assert dev_c.severity == "good"
    assert elite_c.severity in ("minor", "major")


def test_every_check_has_a_source():
    for level in ("developmental", "trained", "elite"):
        for _phase, checks in sprint_checks(level).items():
            for _key, check in checks:
                assert check.source, f"{check.metric} missing source"
        for _attr, check in jump_checks(level):
            assert check.source, f"{check.metric} missing source"


def test_findings_carry_confidence():
    kpts, metrics, vel = _sprint_setup()
    findings = analyze_sprint_form(kpts, metrics, vel, FPS)
    assert all(f.confidence is not None for f in findings)
    contact = next(f for f in findings if f.metric == "Ground contact time")
    # At 100 fps the fixed ~11-frame contact should not be frame-rate limited.
    assert contact.confidence.level in ("High", "Medium")


def test_invalid_level_falls_back_to_trained():
    a = sprint_checks("nonsense")["drive"]
    b = sprint_checks("trained")["drive"]
    assert dict(a)["contact_ms"].hi == dict(b)["contact_ms"].hi

# --- key / deviation / bucketing (diagnostics + radar support) ---------------

def test_findings_carry_key_and_deviation():
    kpts, metrics, vel = _sprint_setup(trunk=30.0)
    findings = analyze_sprint_form(kpts, metrics, vel, FPS)
    trunk = next(f for f in findings if f.metric == "Trunk lean")
    assert trunk.key == "trunk" and trunk.deviation == "high"
    for f in findings:
        if f.severity == "good":
            assert f.deviation == ""
        assert f.key


def test_jump_findings_carry_key_and_deviation():
    findings = analyze_jump_form(_jump_metrics(peak_knee_flexion_landing=155.0))
    landing = next(f for f in findings
                   if f.metric == "Peak knee flexion on landing")
    assert landing.key == "peak_knee_flexion_landing"
    assert landing.deviation == "high"


def test_metric_help_covers_every_sprint_and_jump_key():
    for level in ("developmental", "trained", "elite"):
        for _phase, checks in sprint_checks(level).items():
            for key, _check in checks:
                assert metric_help(key), f"no metric_help() text for {key!r}"
        for attr, _check in jump_checks(level):
            assert metric_help(attr), f"no metric_help() text for {attr!r}"


def test_metric_help_unknown_key_is_empty_not_an_error():
    assert metric_help("not_a_real_key") == ""


def test_bucket_sprint_steps_constant_speed():
    from athletic_analysis.core.coaching import bucket_sprint_steps
    kpts, metrics, vel = _sprint_setup()
    buckets = bucket_sprint_steps(kpts, metrics, vel, FPS)
    assert set(buckets) == {"max velocity"}
    b = buckets["max velocity"]
    assert len(b.strike_frames) == len(metrics.steps)
    assert set(b.values) == {"trunk", "contact_ms", "knee_strike", "thigh",
                             "cadence", "overstride"}
