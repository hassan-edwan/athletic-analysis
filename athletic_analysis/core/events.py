"""Detection of gait and jump events from foot/hip trajectories.

Image coordinates: y grows downward, so the ground is at *high* y and being
airborne means *low* foot y. Thresholds are scaled by each signal's own
amplitude so detection works at any video resolution and framing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from athletic_analysis.core.pose.skeleton import KP


@dataclass
class GaitEvent:
    frame: int
    side: str  # "left" | "right"
    kind: str  # "strike" | "toeoff"


@dataclass
class JumpPhases:
    takeoff_frame: int
    landing_frame: int
    lowest_hip_frame: int  # countermovement bottom (max image-y of hip before takeoff)
    baseline_hip_y: float  # standing hip height in px (image y)


def foot_y_signal(kpts: np.ndarray, side: str) -> np.ndarray:
    """Vertical position of the foot: mean of big toe and heel."""
    prefix = "l_" if side == "left" else "r_"
    toe = kpts[:, KP[prefix + "big_toe"], 1]
    heel = kpts[:, KP[prefix + "heel"], 1]
    return (toe + heel) / 2.0


def contact_mask(foot_y: np.ndarray, fps: float,
                 ground_band: float = 0.15, speed_factor: float = 2.5,
                 min_contact_s: float = 0.04, merge_gap_s: float = 0.03) -> np.ndarray:
    """Boolean mask of frames where this foot is on the ground.

    Contact = foot near the ground level (top `ground_band` fraction of its
    vertical travel) AND moving slowly (|vy| below `speed_factor` amplitudes/s;
    swing-phase foot speed is several times higher).
    """
    y = np.asarray(foot_y, dtype=np.float64)
    n = len(y)
    if n < 3:
        return np.zeros(n, dtype=bool)
    lo, hi = np.nanpercentile(y, 5), np.nanpercentile(y, 95)
    amp = hi - lo
    if amp <= 1e-9:  # foot never moves: treat as always on the ground
        return np.ones(n, dtype=bool)
    ground = np.nanpercentile(y, 90)
    near_ground = y > ground - ground_band * amp
    vy = np.gradient(y) * fps
    slow = np.abs(vy) < speed_factor * amp
    mask = near_ground & slow

    # Merge sub-gap holes, then drop implausibly short contacts.
    mask = _close_gaps(mask, max(1, round(merge_gap_s * fps)))
    mask = _drop_short_runs(mask, max(2, round(min_contact_s * fps)))
    return mask


def contact_threshold(foot_y: np.ndarray, ground_band: float = 0.15) -> float | None:
    """The ground-proximity boundary `contact_mask` uses (foot on the ground
    when foot_y is above it, i.e. numerically greater — y grows downward).
    None when the foot never moves enough to define a threshold."""
    y = np.asarray(foot_y, dtype=np.float64)
    if len(y) < 3:
        return None
    lo, hi = np.nanpercentile(y, 5), np.nanpercentile(y, 95)
    amp = hi - lo
    if amp <= 1e-9:
        return None
    return float(np.nanpercentile(y, 90) - ground_band * amp)


def refine_event_time(foot_y: np.ndarray, frame: int, kind: str,
                      ground_band: float = 0.05) -> float:
    """Fractional frame index of a strike/toe-off, by linearly interpolating
    where foot_y crosses the ground threshold. Falls back to `frame` when the
    geometry doesn't support a crossing (flat foot, clip edge, low confidence).

    This recovers sub-frame precision: `detect_gait_events` can only land on a
    whole frame, and it places a strike where the foot is near-ground *and*
    slow — a few frames *after* the foot first touches on a fast descent. The
    true contact instant is the air→ground crossing, so a strike walks back to
    the first ground frame of its contact run and interpolates that crossing;
    a toe-off walks forward to the ground→air crossing.

    The crossing threshold (`ground_band`, default 0.05) is deliberately
    *tighter* than the detection band (0.15): detection is loose so it reliably
    catches the near-ground stance phase, but for timing we want the edge near
    the actual ground level so the refined contact time tracks true touchdown/
    toe-off (including the loading/push-off transition) rather than a point 15%
    up the swing."""
    y = np.asarray(foot_y, dtype=np.float64)
    n = len(y)
    thr = contact_threshold(foot_y, ground_band)
    if thr is None or not (0 <= frame < n) or not np.isfinite(y[frame]):
        return float(frame)

    def _interp(air_i: int, ground_i: int) -> float:
        y0, y1 = y[air_i], y[ground_i]
        if not (np.isfinite(y0) and np.isfinite(y1)) or y0 == y1:
            return float(min(air_i, ground_i) + (0 if air_i < ground_i else 1))
        frac = float(np.clip((thr - y0) / (y1 - y0), 0.0, 1.0))
        return air_i + frac if air_i < ground_i else ground_i + (1.0 - frac)

    if kind == "strike":
        j = frame  # walk back to the first ground frame of this contact run
        while j > 0 and np.isfinite(y[j - 1]) and y[j - 1] >= thr:
            j -= 1
        if j == 0:
            return 0.0
        return _interp(j - 1, j)      # crossing between air (j-1) and ground (j)
    else:  # toeoff: walk forward to the last ground frame, then the crossing up
        j = frame
        while j < n - 1 and np.isfinite(y[j + 1]) and y[j + 1] >= thr:
            j += 1
        if j == n - 1:
            return float(n - 1)
        return _interp(j + 1, j)      # crossing between ground (j) and air (j+1)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def _close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    inv_runs = _runs(~mask)
    for s, e in inv_runs:
        if s == 0 or e == len(mask):
            continue  # keep leading/trailing gaps
        if e - s <= max_gap:
            out[s:e] = True
    return out


def _drop_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = mask.copy()
    for s, e in _runs(mask):
        if e - s < min_len:
            out[s:e] = False
    return out


def detect_gait_events(kpts: np.ndarray, fps: float) -> list[GaitEvent]:
    """Foot strikes and toe-offs for both feet, sorted by frame."""
    events: list[GaitEvent] = []
    for side in ("left", "right"):
        mask = contact_mask(foot_y_signal(kpts, side), fps)
        for s, e in _runs(mask):
            if s > 0:
                events.append(GaitEvent(frame=int(s), side=side, kind="strike"))
            if e < len(mask):
                events.append(GaitEvent(frame=int(e - 1), side=side, kind="toeoff"))
    events.sort(key=lambda ev: ev.frame)
    return events


def gait_event_anomalies(events: list[GaitEvent], fps: float) -> list[str]:
    """Cheap sanity flags on the *detected* events (not a re-detection): a
    running gait alternates feet with fairly regular step intervals and
    physiologically bounded ground contacts, so violations point at a missed
    or doubled strike. Returned as short human notes for the quality badge —
    detection itself is left unchanged (a full detector rewrite would ripple
    into every metric and the iOS parity fixtures; this just says 'the timeline
    here looks off, eyeball it')."""
    strikes = [ev for ev in events if ev.kind == "strike"]
    notes: list[str] = []
    if len(strikes) < 3:
        return notes

    # 1) Feet should alternate; two same-side strikes in a row => a missed one.
    same_side = sum(1 for a, b in zip(strikes, strikes[1:]) if a.side == b.side)
    if same_side:
        notes.append(f"{same_side} step(s) don't alternate feet — a strike may "
                     "be missed or doubled")

    # 2) Step intervals should be roughly regular; big outliers => missed/doubled.
    intervals = np.diff([s.frame for s in strikes])
    if len(intervals) >= 3:
        med = float(np.median(intervals))
        if med > 0:
            irregular = int(np.sum((intervals > 1.8 * med) | (intervals < 0.55 * med)))
            if irregular:
                notes.append(f"{irregular} step interval(s) look irregular "
                             "(possible missed/doubled strike)")

    # 3) Ground contacts should sit within a plausible band (~40-400 ms).
    odd_contacts = 0
    for s in strikes:
        toeoff = next((ev for ev in events if ev.kind == "toeoff"
                       and ev.side == s.side and ev.frame > s.frame), None)
        if toeoff:
            c = (toeoff.frame - s.frame) / fps
            if c < 0.04 or c > 0.40:
                odd_contacts += 1
    if odd_contacts:
        notes.append(f"{odd_contacts} ground contact(s) outside the usual "
                     "40-400 ms range")
    return notes


def detect_jump(kpts: np.ndarray, fps: float,
                min_flight_s: float = 0.15) -> JumpPhases | None:
    """Find the main jump: the airborne interval (both feet off ground) with the
    greatest hip rise. Returns None if no plausible flight phase exists."""
    left = contact_mask(foot_y_signal(kpts, "left"), fps)
    right = contact_mask(foot_y_signal(kpts, "right"), fps)
    airborne = ~left & ~right
    hip_y = kpts[:, KP["hip_center"], 1].astype(np.float64)

    min_len = max(2, round(min_flight_s * fps))
    baseline_n = max(2, round(0.5 * fps))
    baseline_hip = float(np.nanmedian(hip_y[:baseline_n]))

    best: tuple[float, tuple[int, int]] | None = None
    for s, e in _runs(airborne):
        if e - s < min_len or s == 0 or e >= len(airborne):
            continue
        rise = baseline_hip - float(np.nanmin(hip_y[s:e]))  # px above standing
        if best is None or rise > best[0]:
            best = (rise, (s, e))
    if best is None:
        return None
    s, e = best[1]
    takeoff, landing = int(s - 1), int(e)
    pre = hip_y[:takeoff + 1]
    lowest = int(np.nanargmax(pre)) if len(pre) else takeoff
    return JumpPhases(takeoff_frame=takeoff, landing_frame=landing,
                      lowest_hip_frame=lowest, baseline_hip_y=baseline_hip)
