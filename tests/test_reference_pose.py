import numpy as np
import pytest

from athletic_analysis.core import reference_pose as rp


@pytest.mark.parametrize("key", ["trunk", "knee_strike", "thigh"])
@pytest.mark.parametrize("target", [-10.0, 15.0, 80.0, 145.0, 175.0])
def test_render_produces_a_valid_image_for_every_posable_key(key, target):
    img = rp.render(key, target, "deg")
    assert img.shape == (rp.CANVAS_H, rp.CANVAS_W, 3)
    assert img.dtype == np.uint8
    # Something was actually drawn, not just the blank ground line.
    assert img.std() > 5


def test_render_rejects_non_posable_keys():
    with pytest.raises(ValueError):
        rp.render("contact_ms", 100.0, "ms")


@pytest.mark.parametrize("key", ["trunk", "knee_strike", "thigh"])
def test_render_sequence_returns_valid_frames(key):
    frames = rp.render_sequence(key, 20.0, "deg", n_frames=8)
    assert len(frames) == 8
    for frame in frames:
        assert frame.shape == (rp.CANVAS_H, rp.CANVAS_W, 3)
        assert frame.dtype == np.uint8


def test_render_sequence_settles_on_target():
    # The hold segment (t in [0.35, 0.55]) should render pixel-identical to
    # the plain static render() at exactly the target angle.
    frames = rp.render_sequence("trunk", 20.0, "deg", n_frames=11)
    still = rp.render("trunk", 20.0, "deg")
    # frame index 4 of 11 -> t = 0.4, inside the hold segment.
    assert np.array_equal(frames[4], still)


def test_motion_offset_is_zero_at_holds():
    assert rp._motion_offset(0.35) == 0.0
    assert rp._motion_offset(0.55) == 0.0
    assert rp._motion_offset(1.0) == 0.0
    assert rp._motion_offset(0.0) == -15.0
