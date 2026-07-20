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
