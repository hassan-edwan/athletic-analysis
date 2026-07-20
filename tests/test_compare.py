import numpy as np

from athletic_analysis.core.coaching import analyze_sprint_form, bucket_sprint_steps
from athletic_analysis.core.compare import build_comparisons
from athletic_analysis.core.metrics.sprint import SprintMetrics, StepRecord
from tests.conftest import make_sequence

FPS = 100.0


def _step(frame: int, side: str, *, trunk=6.0, contact=0.11) -> StepRecord:
    return StepRecord(side=side, strike_frame=frame, toeoff_frame=frame + 11,
                      contact_time_s=contact, flight_time_s=0.13,
                      step_time_s=0.24, step_length=0.9, step_speed=3.8,
                      knee_angle_at_strike=155.0, swing_thigh_angle=75.0,
                      trunk_lean_at_strike=trunk)


def _setup(trunks: list[float], contact=0.11):
    T = 300
    kpts = make_sequence(T)
    kpts[:, :, 0] += np.arange(T)[:, None] * 8.0
    metrics = SprintMetrics()
    for i, trunk in enumerate(trunks, start=1):
        metrics.steps.append(_step(25 * i, "left" if i % 2 == 0 else "right",
                                   trunk=trunk, contact=contact))
    velocities = {"run_speed": np.full(T, 800.0)}  # constant => max velocity
    findings = analyze_sprint_form(kpts, metrics, velocities, FPS)
    buckets = bucket_sprint_steps(kpts, metrics, velocities, FPS)
    return findings, buckets


def test_real_best_step_used_when_one_step_is_in_range():
    # Six steps way out of range (30 deg), one right in the middle of the
    # 8-10 deg max-velocity band -> the median finding is still flagged, but
    # a real good step exists and must be preferred over a synthetic figure.
    findings, buckets = _setup([30.0] * 6 + [4.0])
    trunk_finding = next(f for f in findings if f.metric == "Trunk lean")
    assert trunk_finding.severity != "good"
    comparisons = build_comparisons(findings, buckets)
    trunk_cmp = next(c for c in comparisons if c.finding.metric == "Trunk lean")
    assert trunk_cmp.best_frame is not None
    assert trunk_cmp.best_value == 4.0
    assert trunk_cmp.posable


def test_synthetic_fallback_when_every_step_shares_the_fault():
    findings, buckets = _setup([30.0] * 7)
    trunk_finding = next(f for f in findings if f.metric == "Trunk lean")
    assert trunk_finding.severity != "good"
    comparisons = build_comparisons(findings, buckets)
    trunk_cmp = next(c for c in comparisons if c.finding.metric == "Trunk lean")
    assert trunk_cmp.best_frame is None
    assert trunk_cmp.best_value is None
    assert trunk_cmp.posable


def test_timing_metrics_are_never_posable():
    findings, buckets = _setup([6.0] * 7, contact=0.30)  # long contact fault
    comparisons = build_comparisons(findings, buckets)
    contact_cmp = next(c for c in comparisons if c.finding.metric == "Ground contact time")
    assert not contact_cmp.posable


def test_good_findings_produce_no_comparisons():
    findings, buckets = _setup([6.0] * 7)
    assert all(f.severity == "good" for f in findings)
    assert build_comparisons(findings, buckets) == []
