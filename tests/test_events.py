import numpy as np

from athletic_analysis.core.events import (contact_mask, contact_threshold,
                                           detect_gait_events, detect_jump,
                                           refine_event_time)
from athletic_analysis.core.pose.skeleton import KP
from tests.conftest import make_sequence

FPS = 100.0


def _gait_foot_y(T: int, phase_offset: int, ground: float = 600.0,
                 contact_len: int = 15, swing_len: int = 35,
                 swing_lift: float = 100.0) -> np.ndarray:
    """Periodic contact plateau + sinusoidal swing, starting at `phase_offset`."""
    y = np.full(T, ground)
    cycle = contact_len + swing_len
    for t in range(T):
        pos = (t - phase_offset) % cycle
        if pos >= contact_len:
            swing_pos = (pos - contact_len) / swing_len
            y[t] = ground - swing_lift * np.sin(np.pi * swing_pos)
    return y


def _apply_foot_y(kpts: np.ndarray, side: str, y: np.ndarray) -> None:
    prefix = "l_" if side == "left" else "r_"
    for name in (prefix + "big_toe", prefix + "small_toe", prefix + "heel",
                 prefix + "ankle"):
        offset = kpts[0, KP[name], 1] - 600.0
        kpts[:, KP[name], 1] = y + offset


def test_contact_mask_finds_plateaus():
    y = _gait_foot_y(200, phase_offset=0)
    mask = contact_mask(y, FPS)
    # Contacts occur at cycle starts: frames 0-14, 50-64, 100-114, 150-164.
    for start in (55, 105, 155):
        assert mask[start], f"expected contact at frame {start}"
    for mid_swing in (30, 80, 130, 180):
        assert not mask[mid_swing], f"expected swing at frame {mid_swing}"


def test_detect_gait_events_alternating_feet():
    T = 300
    kpts = make_sequence(T)
    _apply_foot_y(kpts, "left", _gait_foot_y(T, phase_offset=0))
    _apply_foot_y(kpts, "right", _gait_foot_y(T, phase_offset=25))
    events = detect_gait_events(kpts, FPS)
    strikes = [ev for ev in events if ev.kind == "strike"]
    assert len(strikes) >= 8
    # Strikes should alternate sides.
    sides = [s.side for s in strikes]
    assert all(a != b for a, b in zip(sides, sides[1:]))
    # Left strikes near multiples of 50, right strikes near 25 + multiples of 50.
    for s in strikes:
        expected_mod = 0 if s.side == "left" else 25
        assert abs((s.frame - expected_mod) % 50) <= 4 or \
               abs(50 - (s.frame - expected_mod) % 50) <= 4


def test_detect_jump():
    T = 300
    kpts = make_sequence(T)
    takeoff, landing = 150, 190  # 0.4 s flight at 100 fps
    # Countermovement: upper body dips before takeoff while feet stay planted.
    cm = np.zeros(T)
    cm[110:takeoff] = 60 * np.sin(np.pi * np.arange(takeoff - 110) / (takeoff - 110))
    flight = np.arange(landing - takeoff)
    rise = np.zeros(T)
    rise[takeoff:landing] = -160 * np.sin(np.pi * flight / (landing - takeoff))  # up = -y
    feet = [KP[n] for n in ("l_big_toe", "r_big_toe", "l_small_toe", "r_small_toe",
                            "l_heel", "r_heel", "l_ankle", "r_ankle")]
    upper = [i for i in range(26) if i not in feet]
    kpts[:, upper, 1] += (cm + rise)[:, None]
    kpts[:, feet, 1] += rise[:, None]  # feet only move during flight
    jump = detect_jump(kpts, FPS)
    assert jump is not None
    assert abs(jump.takeoff_frame - takeoff) <= 4
    assert abs(jump.landing_frame - landing) <= 4
    assert 110 <= jump.lowest_hip_frame <= takeoff


def test_detect_jump_none_when_standing():
    kpts = make_sequence(200)
    assert detect_jump(kpts, FPS) is None


def test_refine_event_time_recovers_known_subframe_crossing():
    # A foot descending linearly from 500 (air) to 700 (ground) crosses any
    # threshold at a precise sub-frame point. contact_threshold uses the 90th
    # percentile minus 15% of amplitude; place a descent so the true crossing
    # falls between integer frames and confirm we recover it.
    y = np.concatenate([np.full(10, 500.0),                 # airborne plateau
                        np.linspace(500.0, 700.0, 11),      # descent (frames 10-20)
                        np.full(10, 700.0)])                # ground plateau
    band = 0.05
    thr = contact_threshold(y, band)
    assert thr is not None
    # The integer strike is the first frame at/above threshold; the fractional
    # time must be earlier (the crossing happened mid-descent) and land where
    # the line actually crosses `thr`.
    mask = contact_mask(y, FPS)
    strike_frame = int(np.argmax(mask))
    t = refine_event_time(y, strike_frame, "strike", band)
    # Reconstruct the exact crossing on the descent segment (slope 20 px/frame).
    exact = 10 + (thr - 500.0) / 20.0
    assert abs(t - exact) < 0.6
    assert t <= strike_frame  # sub-frame time is never after the integer frame


def test_refine_event_time_falls_back_on_flat_foot():
    flat = np.full(50, 600.0)  # foot never moves -> no threshold
    assert refine_event_time(flat, 20, "strike") == 20.0
