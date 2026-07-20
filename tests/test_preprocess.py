import numpy as np

from athletic_analysis.core.preprocess import (Preprocessor, ReframeTracker,
                                               comb_metric, deinterlace,
                                               enhance_contrast, rotate_frame)


def _low_contrast_frame():
    # Values clustered around mid-gray => low RMS contrast.
    rng = np.random.default_rng(0)
    return (120 + rng.normal(0, 4, (120, 160, 3))).clip(0, 255).astype(np.uint8)


def test_enhance_raises_contrast():
    frame = _low_contrast_frame()
    before = frame.std()
    after = enhance_contrast(frame).std()
    assert after > before


def test_rotate_dimensions_and_roundtrip():
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[0, 0] = (255, 255, 255)  # mark top-left
    r90 = rotate_frame(frame, 90)
    assert r90.shape[:2] == (60, 40)
    # 90 then 270 returns to original.
    assert np.array_equal(rotate_frame(r90, 270), frame)


def test_deinterlace_reduces_comb():
    # Build a combed frame: alternating bright/dark rows (interlaced motion).
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[0::2] = 220
    frame[1::2] = 30
    before = comb_metric(frame)
    after = comb_metric(deinterlace(frame))
    assert after < before


def test_reframe_backmaps_to_original_coords():
    # A person box in the lower-right quadrant of a 400x600 frame.
    boxes = [np.array([300.0, 200.0, 380.0, 360.0])] * 10
    tracker = ReframeTracker(boxes=boxes, frame_w=600, frame_h=400,
                             smooth_window=1)
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    # Put a distinctive pixel at the athlete's center in original coords.
    cx, cy = 340, 280
    frame[cy, cx] = (255, 255, 255)
    crop, to_original = tracker.crop_and_map(frame, 5)
    # The crop must contain the athlete and be at least as large as the source
    # region (upscaled), and back-mapping a crop-space point recovers original.
    assert crop.shape[0] >= 1 and crop.shape[1] >= 1
    # Map the crop's own center back and confirm it lands inside the padded box.
    ch, cw = crop.shape[:2]
    mapped = to_original(np.array([[cw / 2, ch / 2]]))[0]
    assert 300 <= mapped[0] <= 380
    assert 200 <= mapped[1] <= 360


def test_reframe_roundtrip_point_precision():
    boxes = [np.array([100.0, 100.0, 200.0, 300.0])] * 6
    tracker = ReframeTracker(boxes=boxes, frame_w=640, frame_h=480,
                             smooth_window=1)
    x0, y0, w, h = tracker._crops[3]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    crop, to_original = tracker.crop_and_map(frame, 3)
    ch, cw = crop.shape[:2]
    # A crop-space corner maps back to the crop rect's original corner.
    mapped = to_original(np.array([[0.0, 0.0], [cw, ch]]))
    assert abs(mapped[0][0] - x0) < 1 and abs(mapped[0][1] - y0) < 1
    assert abs(mapped[1][0] - (x0 + w)) < 1.5


def test_preprocessor_applied_names():
    p = Preprocessor(rotation=90, enhance=True, reframe=True)
    names = p.applied()
    assert "rotate90" in names and "enhance" in names and "reframe" in names
    assert p.needs_detection_pass()
    assert not Preprocessor().needs_detection_pass()


def test_select_tracked_box_locks_onto_subject():
    from athletic_analysis.core.pose.detector import (box_center, largest_box,
                                                      select_tracked_box)
    # Two people: left (our athlete) and right (momentarily taller bystander).
    athlete = np.array([100.0, 100.0, 160.0, 300.0])   # 200 tall
    bystander = np.array([500.0, 80.0, 560.0, 320.0])   # 240 tall (taller)
    boxes = np.stack([athlete, bystander])
    # Largest-box would grab the bystander...
    assert np.allclose(largest_box(boxes), bystander)
    # ...but tracking from near the athlete's last position stays on the athlete.
    picked = select_tracked_box(boxes, box_center(athlete))
    assert np.allclose(picked, athlete)


def test_preprocessor_process_without_reframe_is_identity_coords():
    p = Preprocessor(enhance=True)
    frame = np.full((50, 50, 3), 100, dtype=np.uint8)
    out, to_original = p.process(frame, 0)
    pts = np.array([[10.0, 20.0]])
    assert np.allclose(to_original(pts), pts)  # enhance doesn't move coords
