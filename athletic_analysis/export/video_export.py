"""Annotated MP4 export: skeleton + angle overlays burned into every frame."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2

import numpy as np

from athletic_analysis.core.pose.skeleton import (draw_angle_labels,
                                                  draw_info_text, draw_pose)
from athletic_analysis.core.session import AnalysisSession
from athletic_analysis.core.video_source import VideoSource

OVERLAY_ANGLES = ["knee_l", "knee_r", "hip_l", "hip_r", "trunk_lean"]


def export_annotated_video(session: AnalysisSession, out_path: str | Path,
                           progress_cb: Callable[[int, int], None] | None = None) -> None:
    source = VideoSource(session.video_path)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             source.fps, (source.width, source.height))
    if not writer.isOpened():
        source.close()
        raise IOError(f"Could not open video writer for {out_path}")
    try:
        total = source.frame_count
        for idx, frame in source.iter_frames():
            if session.keypoints is not None and idx < len(session.keypoints):
                kpts = session.keypoints[idx]
                draw_pose(frame, kpts)
                angles_now = {k: float(session.angles[k][idx])
                              for k in OVERLAY_ANGLES if k in session.angles}
                draw_angle_labels(frame, kpts, angles_now)
                speed = session.velocities.get("run_speed")
                if speed is not None and idx < len(speed) and np.isfinite(speed[idx]):
                    draw_info_text(frame,
                                   f"speed {speed[idx]:.2f} {session.velocity_unit}",
                                   row=1)
            writer.write(frame)
            if progress_cb and (idx % 10 == 0 or idx == total - 1):
                progress_cb(idx + 1, total)
    finally:
        writer.release()
        source.close()
