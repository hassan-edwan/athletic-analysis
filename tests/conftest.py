"""Synthetic pose helpers shared by the analysis tests."""

import numpy as np
import pytest

from athletic_analysis.core.pose.skeleton import KP

# A plausible standing figure in a 1280x720 frame (y grows downward).
STANDING = {
    "nose": (300, 105), "l_eye": (295, 100), "r_eye": (305, 100),
    "l_ear": (290, 102), "r_ear": (310, 102),
    "head": (300, 90), "neck": (300, 150),
    "l_shoulder": (285, 155), "r_shoulder": (315, 155),
    "l_elbow": (280, 230), "r_elbow": (320, 230),
    "l_wrist": (278, 300), "r_wrist": (322, 300),
    "hip_center": (300, 330), "l_hip": (285, 330), "r_hip": (315, 330),
    "l_knee": (285, 460), "r_knee": (315, 460),
    "l_ankle": (285, 580), "r_ankle": (315, 580),
    "l_heel": (280, 598), "r_heel": (320, 598),
    "l_big_toe": (300, 600), "r_big_toe": (330, 600),
    "l_small_toe": (295, 600), "r_small_toe": (335, 600),
}


def standing_pose() -> np.ndarray:
    """(26, 3) standing keypoints with full confidence."""
    kpts = np.zeros((26, 3), dtype=np.float64)
    for name, (x, y) in STANDING.items():
        kpts[KP[name]] = (x, y, 1.0)
    return kpts


def make_sequence(T: int) -> np.ndarray:
    """(T, 26, 3): the standing pose repeated."""
    return np.tile(standing_pose(), (T, 1, 1))


@pytest.fixture
def pose_seq():
    return make_sequence(100)
