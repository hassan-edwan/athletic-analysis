import math

import numpy as np

from athletic_analysis.core import form_overlay as fo
from athletic_analysis.core.angles import compute_angles
from athletic_analysis.core.pose.skeleton import KP

STAND = {
    "l_shoulder": (285, 155), "r_shoulder": (315, 155),
    "l_hip": (285, 330), "r_hip": (315, 330),
    "l_knee": (285, 460), "r_knee": (315, 460),
    "l_ankle": (285, 580), "r_ankle": (300, 560),
    "l_big_toe": (300, 600), "r_big_toe": (315, 580),
    "neck": (300, 150), "head": (300, 120), "hip_center": (300, 330),
    "l_heel": (280, 598), "r_heel": (295, 578),
}


def _pose():
    kp = np.zeros((1, 26, 3))
    for n, (x, y) in STAND.items():
        kp[0, KP[n]] = (x, y, 1.0)
    return kp


def _measure_trunk(spec, direction=1.0):
    ax, ay = spec.anchor
    ox, oy = spec.optimal_end
    return math.degrees(math.atan2((ox - ax) * direction, ay - oy))


def test_trunk_corrected_segment_hits_target_angle():
    kp = _pose()
    spec = fo._trunk_spec(kp[0], 1.0, 32.0)
    assert abs(_measure_trunk(spec) - 32.0) < 0.01


def test_corrected_segment_coincides_when_target_equals_actual():
    kp = _pose()
    val = float(compute_angles(kp)["trunk_lean"][0])
    spec = fo._trunk_spec(kp[0], 1.0, val)
    assert np.allclose(spec.actual_end, spec.optimal_end, atol=0.5)


def test_knee_corrected_shank_hits_interior_angle():
    kp = _pose()
    spec = fo._knee_spec(kp[0], "l", 140.0)
    knee = np.array(spec.anchor)
    hip = kp[0, KP["l_hip"], :2]
    opt = np.array(spec.optimal_end)
    v1, v2 = hip - knee, opt - knee
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    assert abs(math.degrees(math.acos(np.clip(cos, -1, 1))) - 140.0) < 0.1


def test_stance_and_swing_sides_are_opposite():
    kp = _pose()  # r_ankle y=560 (higher up) -> right is swing, left is stance
    assert fo.stance_side(kp[0]) == "l"
    assert fo.swing_side(kp[0]) == "r"


def test_target_angle_is_nearest_band_edge():
    assert fo._target_angle(5.0, 8.0, 10.0) == 8.0     # below -> lo
    assert fo._target_angle(30.0, 8.0, 10.0) == 10.0   # above -> hi
    assert fo._target_angle(9.0, 8.0, 10.0) == 9.0     # in range -> unchanged


def test_live_eval_flags_major_trunk_lean_and_builds_spec():
    # A big forward lean at max velocity is a major fault; a spec must exist.
    kp = np.tile(_pose(), (5, 1, 1))
    # tip the shoulders forward (travel +x) so trunk leans ~30 deg
    for f in range(5):
        kp[f, KP["l_shoulder"]] = (285 + 100, 200, 1.0)
        kp[f, KP["r_shoulder"]] = (315 + 100, 200, 1.0)
    angles = compute_angles(kp)
    spans = [(0, 4, "max velocity")]
    ev = fo.live_eval("trunk", 2, kp, angles, spans, "trained", 1.0)
    assert ev is not None
    assert ev.severity == "major"
    assert ev.off > 0 and ev.over is True
    assert ev.spec is not None


def test_live_eval_none_outside_graded_phase():
    kp = np.tile(_pose(), (3, 1, 1))
    angles = compute_angles(kp)
    ev = fo.live_eval("trunk", 1, kp, angles, [(0, 2, "deceleration")],
                      "trained", 1.0)
    assert ev is None  # deceleration has no checks
