import numpy as np

from athletic_analysis.core.coaching import FormFinding, analyze_sprint_form
from athletic_analysis.core.diagnostics import all_diagnoses, diagnose
from tests.test_coaching import FPS, _sprint_setup


def _fault(key: str, deviation: str, phase: str = "max velocity") -> FormFinding:
    return FormFinding(phase=phase, metric=key, value=0.0, value_text="",
                       target_text="", severity="major", cue="", frame=0,
                       key=key, deviation=deviation)


def test_every_diagnosis_is_complete():
    for d in all_diagnoses():
        assert d.title, d
        assert d.technical_causes and all(d.technical_causes), d.key
        assert d.muscle_factors and all(d.muscle_factors), d.key
        assert d.drills and all(d.drills), d.key
        assert d.source, d.key


def test_every_genuine_fault_has_a_diagnosis():
    # (key, deviation, phase) pairs that represent real faults, not
    # tracking/measurement artifacts.
    faults = [
        ("overstride", "high", "max velocity"),
        ("contact_ms", "high", "drive"),
        ("contact_ms", "high", "max velocity"),
        ("contact_ms", "low", "max velocity"),
        ("knee_strike", "low", "acceleration"),
        ("knee_strike", "high", "max velocity"),
        ("thigh", "low", "max velocity"),
        ("cadence", "low", "max velocity"),
        ("cadence", "high", "max velocity"),
        ("trunk", "low", "drive"),
        ("trunk", "high", "drive"),
        ("trunk", "low", "acceleration"),
        ("trunk", "high", "acceleration"),
        ("trunk", "low", "max velocity"),
        ("trunk", "high", "max velocity"),
    ]
    for key, deviation, phase in faults:
        assert diagnose(_fault(key, deviation, phase)) is not None, \
            (key, deviation, phase)


def test_artifact_directions_have_no_diagnosis():
    assert diagnose(_fault("overstride", "low")) is None
    assert diagnose(_fault("thigh", "high")) is None


def test_good_and_unknown_return_none():
    good = _fault("trunk", "")
    good.severity = "good"
    assert diagnose(good) is None
    assert diagnose(_fault("", "high")) is None
    # Jump metric keys have no sprint entries.
    assert diagnose(_fault("peak_knee_flexion_landing", "high", "jump")) is None


def test_trunk_diagnosis_is_phase_specific():
    drive = diagnose(_fault("trunk", "low", "drive"))
    maxv = diagnose(_fault("trunk", "low", "max velocity"))
    assert drive is not None and maxv is not None
    assert drive.title != maxv.title


def test_overstride_diagnosis_content():
    d = diagnose(_fault("overstride", "high"))
    muscles = " ".join(d.muscle_factors).lower()
    drills = " ".join(d.drills).lower()
    assert "hamstring" in muscles and "glute" in muscles
    assert "a-skip" in drills and "nordic" in drills


def test_end_to_end_finding_diagnoses():
    kpts, metrics, vel = _sprint_setup(trunk=30.0)
    findings = analyze_sprint_form(kpts, metrics, vel, FPS)
    trunk = next(f for f in findings if f.key == "trunk")
    assert trunk.deviation == "high"
    d = diagnose(trunk)
    assert d is not None and d.technical_causes
