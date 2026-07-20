import cv2
import numpy as np
import pytest

from athletic_analysis.core.assessment import (assess_video, blur_score,
                                               brightness, rms_contrast)


class FakeDetector:
    """Returns a fixed set of boxes for every frame (bypasses the model)."""

    def __init__(self, boxes):
        self._boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    def detect(self, frame):
        return self._boxes


@pytest.fixture
def clip_factory(tmp_path):
    def make(name, frames, fps=30.0):
        path = str(tmp_path / name)
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        return path
    return make


def test_image_metrics_basic():
    dark = np.full((60, 80, 3), 20, dtype=np.uint8)
    bright = np.full((60, 80, 3), 180, dtype=np.uint8)
    assert brightness(dark) < brightness(bright)
    flat = np.full((60, 80, 3), 128, dtype=np.uint8)
    noisy = (np.random.default_rng(0).integers(0, 255, (60, 80, 3))).astype(np.uint8)
    assert rms_contrast(noisy) > rms_contrast(flat)


def test_blur_score_lower_for_blurred():
    rng = np.random.default_rng(1)
    sharp = rng.integers(0, 255, (80, 80, 3)).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (9, 9), 5)
    assert blur_score(blurred) < blur_score(sharp)


def test_low_fps_flagged(clip_factory):
    frames = [np.full((240, 320, 3), 128, dtype=np.uint8) for _ in range(8)]
    path = clip_factory("lowfps.mp4", frames, fps=30.0)
    a = assess_video(path, detector=FakeDetector([[130, 40, 190, 220]]), sample=6)
    assert any("frame rate" in i.title.lower() for i in a.issues)


def test_small_athlete_flags_reframe(clip_factory):
    # Frame 400 tall; person box only ~60 px tall => ~15% fill.
    frames = [np.full((400, 300, 3), 128, dtype=np.uint8) for _ in range(6)]
    path = clip_factory("small.mp4", frames, fps=120.0)
    a = assess_video(path, detector=FakeDetector([[140, 170, 175, 230]]), sample=6)
    assert "reframe" in a.recommended_transforms
    assert any("small" in i.title.lower() for i in a.issues)


def test_multi_panel_flags_reframe(clip_factory):
    frames = [np.full((300, 600, 3), 128, dtype=np.uint8) for _ in range(6)]
    path = clip_factory("multi.mp4", frames, fps=120.0)
    # Two well-separated people (left and right halves).
    det = FakeDetector([[30, 60, 120, 260], [420, 60, 520, 260]])
    a = assess_video(path, detector=det, sample=6)
    assert "reframe" in a.recommended_transforms
    assert any("multiple" in i.title.lower() or "split" in i.title.lower()
               for i in a.issues)


def test_dark_clip_flags_enhance(clip_factory):
    frames = [np.full((300, 300, 3), 25, dtype=np.uint8) for _ in range(6)]
    path = clip_factory("dark.mp4", frames, fps=120.0)
    a = assess_video(path, detector=FakeDetector([[100, 30, 200, 280]]), sample=6)
    assert "enhance" in a.recommended_transforms


def test_good_clip_grades_good(clip_factory):
    rng = np.random.default_rng(2)
    frames = [rng.integers(40, 210, (600, 400, 3)).astype(np.uint8)
              for _ in range(6)]
    path = clip_factory("good.mp4", frames, fps=120.0)
    # Big, single, centered athlete.
    a = assess_video(path, detector=FakeDetector([[120, 40, 280, 560]]), sample=6)
    assert a.grade in ("Good", "Fair")
    assert not any(i.severity == "major" for i in a.issues)
