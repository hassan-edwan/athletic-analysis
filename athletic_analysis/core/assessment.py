"""Fast pre-flight assessment of how suitable a video is for pose analysis.

Samples a handful of frames (not the whole clip), combines container metadata,
a person-detection pass, and cheap image metrics into an overall grade plus
itemized issues. Each issue may recommend a preprocessing transform so the app
can auto-apply fixes. Runs in seconds, before the expensive full pose pass.

Heuristic and sampled by design — it can miss problems in unsampled frames, so
it is presented as guidance, not a guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from athletic_analysis.core.preprocess import comb_metric
from athletic_analysis.core.video_source import VideoSource

GOOD, FAIR, POOR = "Good", "Fair", "Poor"
_GRADE_RANK = {GOOD: 0, FAIR: 1, POOR: 2}


@dataclass
class Issue:
    severity: str          # "minor" | "major"
    title: str
    detail: str
    suggested_transform: str = ""  # "reframe" | "enhance" | "deinterlace" |
    #                                "rotate" | "" (advisory only)


@dataclass
class VideoAssessment:
    grade: str = GOOD
    issues: list[Issue] = field(default_factory=list)
    # Convenience for the UI: transforms to pre-check.
    recommended_transforms: list[str] = field(default_factory=list)
    fps: float = 0.0
    width: int = 0
    height: int = 0
    athlete_fill: float = float("nan")  # median bbox height / frame height
    people_seen: int = 0
    rotation_suggestion: int = 0

    def summary(self) -> str:
        n = len(self.issues)
        return (f"Suitability: {self.grade}"
                + (f" · {n} issue{'s' if n != 1 else ''}" if n else " · no issues"))


# --- image metrics (model-free) ----------------------------------------------

def brightness(frame: np.ndarray) -> float:
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def rms_contrast(frame: np.ndarray) -> float:
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).std())


def blur_score(frame: np.ndarray, box: np.ndarray | None = None) -> float:
    """Variance of Laplacian (low = blurry). Restricted to the person box when
    available, so a busy background doesn't mask a blurry athlete."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        roi = gray[max(0, y1):y2, max(0, x1):x2]
        if roi.size > 100:
            gray = roi
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _disjoint_regions(boxes: np.ndarray, frame_w: int) -> int:
    """Count horizontally separated person clusters — a multi-panel signal."""
    if len(boxes) < 2:
        return len(boxes)
    centers = np.sort((boxes[:, 0] + boxes[:, 2]) / 2)
    gaps = np.diff(centers)
    return 1 + int(np.sum(gaps > 0.25 * frame_w))


def assess_video(path: str, detector=None, sample: int = 12) -> VideoAssessment:
    """Sample frames and produce a suitability assessment. `detector` is a
    PersonDetector (injected so callers can reuse one / run off-thread)."""
    src = VideoSource(path)
    a = VideoAssessment(fps=src.fps, width=src.width, height=src.height)
    n = src.frame_count if src.frame_count > 0 else sample
    idxs = np.linspace(0, max(0, n - 1), min(sample, max(1, n))).astype(int)

    brights, contrasts, blurs, combs = [], [], [], []
    fills, region_counts, people_counts = [], [], []
    landscape_person_in_portrait = 0
    for i in idxs:
        frame = src.get_frame(int(i))
        if frame is None:
            continue
        brights.append(brightness(frame))
        contrasts.append(rms_contrast(frame))
        combs.append(comb_metric(frame))
        primary = None
        if detector is not None:
            boxes = detector.detect(frame)
            people_counts.append(len(boxes))
            if len(boxes):
                region_counts.append(_disjoint_regions(boxes, src.width))
                heights = boxes[:, 3] - boxes[:, 1]
                primary = boxes[int(np.argmax(heights))]
                fills.append(float(heights.max()) / src.height)
                pw = primary[2] - primary[0]
                ph = primary[3] - primary[1]
                if src.height > src.width and pw > ph:
                    landscape_person_in_portrait += 1
        blurs.append(blur_score(frame, primary))
    src.close()

    def med(v):
        return float(np.median(v)) if v else float("nan")

    a.width, a.height = src.width, src.height
    a.athlete_fill = med(fills)
    a.people_seen = int(np.median(people_counts)) if people_counts else 0

    # --- frame rate / resolution (advisory) ---
    if src.fps and src.fps < 60:
        a.issues.append(Issue("major" if src.fps < 45 else "minor",
            "Low frame rate",
            f"{src.fps:.0f} fps — contact/flight timing is coarse "
            f"(±{1000 / src.fps:.0f} ms per frame). Film at 60–240 fps.",
            ""))
    if min(src.width, src.height) < 480:
        a.issues.append(Issue("minor", "Low resolution",
            f"{src.width}×{src.height} — fine detail is limited.", "reframe"))

    # --- athlete size / framing ---
    if np.isfinite(a.athlete_fill) and a.athlete_fill < 0.45:
        sev = "major" if a.athlete_fill < 0.3 else "minor"
        a.issues.append(Issue(sev, "Athlete small in frame",
            f"Athlete fills ~{a.athlete_fill * 100:.0f}% of the frame height. "
            "Auto-reframe crops and upscales them for better keypoints.",
            "reframe"))

    # --- multiple people / multi-panel ---
    median_regions = int(np.median(region_counts)) if region_counts else 1
    if median_regions >= 2 or a.people_seen >= 2:
        a.issues.append(Issue("major", "Multiple people / split view",
            f"~{max(median_regions, a.people_seen)} people or panels detected. "
            "Auto-reframe isolates the primary athlete.", "reframe"))

    # --- lighting / contrast ---
    mb, mc = med(brights), med(contrasts)
    if (np.isfinite(mb) and (mb < 60 or mb > 200)) or (np.isfinite(mc) and mc < 40):
        a.issues.append(Issue("minor", "Poor lighting / low contrast",
            f"brightness {mb:.0f}/255, contrast {mc:.0f} — enhancement helps the "
            "detector find the athlete.", "enhance"))

    # --- blur (advisory) ---
    if blurs and med(blurs) < 60:
        a.issues.append(Issue("minor", "Motion blur / soft focus",
            "Limbs look blurred — keypoints on fast segments may wander. "
            "Use a faster shutter if possible.", ""))

    # --- interlacing ---
    if combs and med(combs) > 0.35:
        a.issues.append(Issue("minor", "Interlacing artifacts",
            "Comb artifacts on moving edges detected. Deinterlace to clean them.",
            "deinterlace"))

    # --- orientation ---
    if src.height > src.width and landscape_person_in_portrait >= max(1, len(fills) // 2):
        a.rotation_suggestion = 90
        a.issues.append(Issue("major", "Video may be rotated",
            "Portrait video with a sideways athlete — rotate to upright.",
            "rotate"))

    # --- aggregate grade + recommendations ---
    majors = sum(1 for i in a.issues if i.severity == "major")
    minors = sum(1 for i in a.issues if i.severity == "minor")
    a.grade = POOR if majors >= 2 else FAIR if (majors or minors >= 2) else GOOD
    seen: list[str] = []
    for issue in a.issues:
        t = issue.suggested_transform
        if t and t not in seen:
            seen.append(t)
    a.recommended_transforms = seen
    return a
