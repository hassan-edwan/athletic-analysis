import cv2
import numpy as np
import pytest

from athletic_analysis.core.calibration import Calibration
from athletic_analysis.core.session import AnalysisSession
from athletic_analysis.core.video_source import VideoSource
from tests.conftest import make_sequence


@pytest.fixture
def tiny_video(tmp_path):
    """20-frame video where each frame's mean brightness encodes its index."""
    path = str(tmp_path / "tiny.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    assert writer.isOpened()
    for i in range(20):
        frame = np.full((48, 64, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _frame_index(frame: np.ndarray) -> int:
    return round(float(frame.mean()) / 10)


def test_sequential_and_backward_reads(tiny_video):
    src = VideoSource(tiny_video)
    assert src.frame_count == 20
    assert _frame_index(src.get_frame(0)) == 0
    assert _frame_index(src.get_frame(5)) == 5
    assert _frame_index(src.get_frame(19)) == 19
    # Backward jump must still return the exact frame.
    assert _frame_index(src.get_frame(3)) == 3
    src.close()


def test_iter_frames_complete(tiny_video):
    src = VideoSource(tiny_video)
    frames = list(src.iter_frames())
    assert len(frames) == 20
    assert [idx for idx, _ in frames] == list(range(20))
    src.close()


def test_session_save_load_roundtrip(tiny_video):
    session = AnalysisSession(video_path=tiny_video, fps=100.0, mode="jump")
    session.keypoints_raw = make_sequence(50).astype(np.float32)
    session.calibration = Calibration(meters_per_pixel=0.004)
    session.recompute()
    session.save()

    loaded = AnalysisSession.load(tiny_video, fps=30.0)
    assert loaded is not None
    assert loaded.mode == "jump"
    assert loaded.fps == 100.0
    assert loaded.calibration.meters_per_pixel == pytest.approx(0.004)
    assert loaded.keypoints_raw.shape == (50, 26, 3)
    assert loaded.angles  # pipeline recomputed on load


def test_session_load_missing_returns_none(tmp_path):
    assert AnalysisSession.load(str(tmp_path / "nope.mp4"), fps=30.0) is None


def test_session_v2_roundtrips_level_and_model(tiny_video):
    session = AnalysisSession(video_path=tiny_video, fps=100.0, mode="sprint",
                             athlete_level="elite", model_tier="Accurate")
    session.keypoints_raw = make_sequence(50).astype(np.float32)
    session.recompute()
    session.save()

    loaded = AnalysisSession.load(tiny_video, fps=30.0)
    assert loaded is not None
    assert loaded.athlete_level == "elite"
    assert loaded.model_tier == "Accurate"
    assert loaded.quality is not None


def test_session_loads_old_v1_sidecar_with_defaults(tiny_video, tmp_path):
    import json
    from athletic_analysis.core.session import AnalysisSession as S
    from tests.conftest import make_sequence as mk
    raw = mk(30).astype(np.float32)
    # Simulate a v1 file: no athlete_level / model_tier / rotation keys.
    v1 = {"version": 1, "fps": 60.0, "mode": "jump",
          "meters_per_pixel": None,
          "keypoints_raw": np.round(raw, 2).tolist()}
    S.sidecar_path(tiny_video).write_text(json.dumps(v1), encoding="utf-8")
    loaded = S.load(tiny_video, fps=30.0)
    assert loaded is not None
    assert loaded.athlete_level == "trained"  # default
    assert loaded.model_tier == "Balanced"  # default
    assert loaded.rotation == 0  # default
    assert loaded.transforms == []  # default
    assert loaded.mode == "jump"


def test_session_v3_roundtrips_rotation_and_transforms(tiny_video):
    session = AnalysisSession(video_path=tiny_video, fps=120.0, mode="sprint",
                             rotation=90, transforms=["reframe", "enhance"])
    session.keypoints_raw = make_sequence(40).astype(np.float32)
    session.recompute()
    session.save()
    loaded = AnalysisSession.load(tiny_video, fps=30.0)
    assert loaded is not None
    assert loaded.rotation == 90
    assert loaded.transforms == ["reframe", "enhance"]


def test_video_source_rotation_transposes_dimensions(tiny_video):
    normal = VideoSource(tiny_video)
    rotated = VideoSource(tiny_video, rotation=90)
    assert (rotated.width, rotated.height) == (normal.height, normal.width)
    f = rotated.get_frame(0)
    assert f is not None and f.shape[0] == rotated.height
    normal.close()
    rotated.close()
