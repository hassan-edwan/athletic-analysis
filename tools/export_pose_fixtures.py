"""Export parity fixtures for the Swift pose pre/post-processing port
(PoseProcessing.swift) by running rtmlib's real functions on random data.
No ONNX models are needed — this exercises the pure-math paths only:

- RTMPose crop spec: bbox -> center/scale/warp matrix (bbox_xyxy2cs +
  top_down_affine's aspect fix + get_warp_matrix via cv2.getAffineTransform)
- SimCC decode + coordinate mapping (get_simcc_maximum + RTMPose.postprocess)
- YOLOX raw-output decode + NMS (YOLOX.postprocess on a synthetic tensor)
- Person selection (athletic_analysis rtmpose_backend.select_person)

Regenerate with: python tools/export_pose_fixtures.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rtmlib.tools.object_detection.yolox import YOLOX  # noqa: E402
from rtmlib.tools.pose_estimation.post_processings import \
    get_simcc_maximum  # noqa: E402
from rtmlib.tools.pose_estimation.pre_processings import (bbox_xyxy2cs,  # noqa: E402
                                                          get_warp_matrix)

from athletic_analysis.core.pose.rtmpose_backend import select_person  # noqa: E402

OUT = (REPO / "ios" / "AthleticAnalysisCore" / "Tests"
       / "AthleticAnalysisCoreTests" / "Fixtures" / "pose_processing.json")

rng = np.random.default_rng(7)


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not math.isfinite(f) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def crop_spec_cases():
    """bbox -> center / aspect-fixed padded scale / 2x3 warp matrix."""
    cases = []
    for w, h in ((192, 256), (288, 384)):
        for _ in range(6):
            x1, y1 = rng.uniform(0, 800, 2)
            bw, bh = rng.uniform(40, 700, 2)
            bbox = np.array([x1, y1, x1 + bw, y1 + bh])
            center, scale = bbox_xyxy2cs(bbox, padding=1.25)
            # aspect fix from top_down_affine
            aspect = w / h
            sw, sh = scale
            if sw > sh * aspect:
                scale_fixed = np.array([sw, sw / aspect])
            else:
                scale_fixed = np.array([sh * aspect, sh])
            warp = get_warp_matrix(center, scale_fixed, 0, output_size=(w, h))
            cases.append({
                "bbox": bbox, "input_w": w, "input_h": h,
                "center": center, "scale": scale_fixed,
                "warp": warp,  # 2x3
            })
    return cases


def simcc_cases():
    """Random SimCC tensors -> decoded keypoints in image coordinates."""
    cases = []
    for w, h in ((192, 256), (288, 384)):
        K, Wx, Wy = 26, w * 2, h * 2
        simcc_x = rng.normal(size=(1, K, Wx)).astype(np.float32)
        simcc_y = rng.normal(size=(1, K, Wy)).astype(np.float32)
        # Force a couple of keypoints to the "no detection" branch (vals <= 0).
        simcc_x[0, 0], simcc_y[0, 0] = -1.0, -1.0
        simcc_x[0, 5], simcc_y[0, 5] = -0.5, -0.5

        bbox = np.array([100.0, 50.0, 400.0, 700.0])
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)
        aspect = w / h
        sw, sh = scale
        scale_fixed = (np.array([sw, sw / aspect]) if sw > sh * aspect
                       else np.array([sh * aspect, sh]))

        locs, vals = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / 2.0
        keypoints = keypoints / np.array([w, h]) * scale_fixed
        keypoints = keypoints + center - scale_fixed / 2

        cases.append({
            "input_w": w, "input_h": h, "bbox": bbox,
            "simcc_x": simcc_x[0], "simcc_y": simcc_y[0],
            "expected_kpts": keypoints[0], "expected_scores": vals[0],
        })
    return cases


def yolox_cases():
    """Synthetic raw YOLOX outputs -> boxes via the real rtmlib postprocess."""
    det = YOLOX.__new__(YOLOX)  # skip __init__ (no model load)
    det.model_input_size = (640, 640)
    det.nms_thr = 0.45
    det.score_thr = 0.7
    det.mode = "human"

    cases = []
    for seed in (1, 2):
        r = np.random.default_rng(seed)
        n = 6400 + 1600 + 400
        cols = 85  # 4 box + 1 obj + 80 classes
        raw = r.normal(scale=0.5, size=(1, n, cols)).astype(np.float32)
        raw[..., 4:] = r.uniform(0, 0.4, size=(1, n, 81))
        # Plant a handful of confident, person-class detections.
        for i in r.integers(0, n, size=8):
            raw[0, i, 4] = r.uniform(0.85, 0.99)   # objectness
            raw[0, i, 5] = r.uniform(0.85, 0.99)   # person class
            raw[0, i, 2:4] = r.uniform(1.5, 3.0, 2)  # sizeable boxes
        ratio = 0.5
        input_copy = raw.copy()
        boxes = det.postprocess(raw.copy(), ratio)
        cases.append({
            "input_w": 640, "input_h": 640, "ratio": ratio,
            "raw": input_copy[0], "cols": cols,
            "expected_boxes": np.asarray(boxes, dtype=np.float64),
        })
    return cases


def selection_cases():
    """Person selection across confidence/tracking scenarios."""
    cases = []
    r = np.random.default_rng(3)
    for last in (None, [300.0, 400.0]):
        for _ in range(3):
            n_people = int(r.integers(1, 4))
            kpts = r.uniform(0, 1000, size=(n_people, 26, 2))
            scores = r.uniform(0.1, 0.95, size=(n_people, 26))
            best = select_person(kpts, scores,
                                 None if last is None else np.array(last),
                                 min_conf=0.35, img_diag=float(np.hypot(720, 1280)))
            cases.append({
                "keypoints": kpts, "scores": scores,
                "last_center": last, "min_conf": 0.35,
                "img_diag": float(np.hypot(720, 1280)),
                "expected_index": -1 if best is None else int(best),
            })
    return cases


def main() -> None:
    data = {
        "crop_specs": crop_spec_cases(),
        "simcc": simcc_cases(),
        "yolox": yolox_cases(),
        "selection": selection_cases(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_clean(data)), encoding="utf-8")
    counts = {k: len(v) for k, v in data.items()}
    boxes = sum(len(c["expected_boxes"]) for c in data["yolox"])
    print(f"wrote {OUT.relative_to(REPO)} — {counts}, yolox boxes kept: {boxes}")


if __name__ == "__main__":
    main()
