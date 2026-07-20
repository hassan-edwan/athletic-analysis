"""Main application window: wiring between video, analysis, and panels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (QComboBox, QDockWidget, QFileDialog,
                               QInputDialog, QLabel, QMainWindow, QMessageBox,
                               QProgressBar, QStyle, QTabWidget, QVBoxLayout,
                               QWidget)

from athletic_analysis.core.calibration import Calibration
from athletic_analysis.core.coaching import (ATHLETE_LEVELS, bucket_sprint_steps,
                                             plot_target_bands, segment_phases)
from athletic_analysis.core.compare import build_comparisons
from athletic_analysis.core.settings import MODEL_TIERS, Settings
from athletic_analysis.core.pose.skeleton import (draw_angle_labels,
                                                  draw_info_text, draw_pose)
from athletic_analysis.core.session import AnalysisSession
from athletic_analysis.core.video_source import VideoSource
from athletic_analysis.export.csv_export import export_frames_csv, export_metrics_csv
from athletic_analysis.export.video_export import export_annotated_video
from athletic_analysis.core.preprocess import Preprocessor
from athletic_analysis.ui import theme
from athletic_analysis.ui.analysis_worker import AnalysisWorker
from athletic_analysis.ui.assessment_worker import AssessmentWorker
from athletic_analysis.ui.compare_panel import ComparePanel
from athletic_analysis.ui.form_panel import FormPanel
from athletic_analysis.ui.keyframe_strip import KeyframeStrip
from athletic_analysis.ui.metrics_panel import MetricsPanel
from athletic_analysis.ui.rep_card import RepCard
from athletic_analysis.ui.step_charts import StepCharts
from athletic_analysis.ui.suitability_panel import SuitabilityPanel
from athletic_analysis.ui.plot_panel import PlotPanel
from athletic_analysis.ui.timeline import Timeline
from athletic_analysis.ui.transport import Transport
from athletic_analysis.ui.video_widget import VideoWidget

OVERLAY_ANGLES = ["knee_l", "knee_r", "hip_l", "hip_r", "trunk_lean"]

# Left/right use the same theme colors as the video overlay skeleton and
# every chart (see theme.py's docstring — this used to be a real bug: the
# right-side marker color here was a BGR tuple copied into an RGB QColor).
MARKER_COLORS = {
    ("strike", "left"): theme.qcolor(theme.LEG_LEFT),
    ("strike", "right"): theme.qcolor(theme.LEG_RIGHT),
    ("toeoff", "left"): theme.qcolor(theme.LEG_LEFT_DARK),
    ("toeoff", "right"): theme.qcolor(theme.LEG_RIGHT_DARK),
    "takeoff": theme.qcolor(theme.TAKEOFF),
    "landing": theme.qcolor(theme.LANDING_EVENT),
    "cm_bottom": theme.qcolor(theme.CM_BOTTOM),
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Athletic Analysis")
        self.resize(1400, 900)

        self.source: VideoSource | None = None
        self.session: AnalysisSession | None = None
        self.current_frame = 0
        self._worker: AnalysisWorker | None = None
        self._assessor: AssessmentWorker | None = None
        self._pick_points: list[tuple[float, float]] = []
        self._video_path: str = ""
        self._pending_transforms: list[str] = []
        self.settings = Settings.load()

        # central: video + transport + timeline
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        self.video = VideoWidget()
        self.transport = Transport()
        self.timeline = Timeline()
        layout.addWidget(self.video, stretch=1)
        layout.addWidget(self.transport)
        layout.addWidget(self.timeline)
        self.setCentralWidget(central)

        # Single side panel, single tab strip — previously this was two
        # separate tabified dock groups (right + bottom) whose tab bars
        # collided at the shared corner and squeezed each other to slivers.
        # One QTabWidget in one dock means one row of tabs, full dock width,
        # and a natural progressive order: check the clip, see the verdict,
        # dig into why, then the raw numbers underneath.
        self.suitability_panel = SuitabilityPanel()
        self.rep_card = RepCard()
        self.form_panel = FormPanel(self._render_replay)
        self.compare_panel = ComparePanel(self._render_replay)
        self.step_charts = StepCharts()
        self.keyframe_strip = KeyframeStrip()
        self.plot_panel = PlotPanel()
        self.metrics_panel = MetricsPanel()

        side_tabs = QTabWidget()
        side_tabs.addTab(self.suitability_panel, "Suitability")
        side_tabs.addTab(self.rep_card, "Summary")
        side_tabs.addTab(self.form_panel, "Form")
        side_tabs.addTab(self.compare_panel, "Compare")
        side_tabs.addTab(self.step_charts, "Steps")
        side_tabs.addTab(self.keyframe_strip, "Key frames")
        side_tabs.addTab(self.plot_panel, "Curves")
        side_tabs.addTab(self.metrics_panel, "Metrics")
        self.side_tabs = side_tabs

        dock_side = QDockWidget("Analysis", self)
        dock_side.setObjectName("dock_side")
        dock_side.setWidget(side_tabs)
        dock_side.setMinimumWidth(440)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_side)

        # toolbar
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        style = self.style()
        self.act_open = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),
            "Open Video…", self)
        self.act_analyze = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
            "Run Pose Analysis", self)
        self.act_capture_fps = QAction("Capture FPS…", self)
        self.act_calibrate = QAction("Calibrate…", self)
        self.act_export_csv = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "Export CSV…", self)
        self.act_export_video = QAction(
            style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "Export Annotated Video…", self)
        toolbar.addAction(self.act_open)
        toolbar.addAction(self.act_analyze)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Mode: "))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Sprint", "sprint")
        self.mode_combo.addItem("Jump", "jump")
        toolbar.addWidget(self.mode_combo)
        toolbar.addWidget(QLabel(" Level: "))
        self.level_combo = QComboBox()
        for lvl in ATHLETE_LEVELS:
            self.level_combo.addItem(lvl.capitalize(), lvl)
        self.level_combo.setToolTip(
            "Athlete level — shifts the optimal ranges (e.g. elite ground "
            "contact ~90 ms vs developmental ~150 ms). Re-grades instantly.")
        toolbar.addWidget(self.level_combo)
        toolbar.addWidget(QLabel(" Model: "))
        self.model_combo = QComboBox()
        for tier in MODEL_TIERS:
            self.model_combo.addItem(tier, tier)
        self.model_combo.setToolTip(
            "Pose model tier. Accurate is more precise but downloads a larger "
            "model once and runs slower on CPU. Takes effect on the next "
            "'Run Pose Analysis'.")
        toolbar.addWidget(self.model_combo)
        toolbar.addSeparator()
        toolbar.addAction(self.act_capture_fps)
        toolbar.addAction(self.act_calibrate)
        toolbar.addAction(self.act_export_csv)
        toolbar.addAction(self.act_export_video)

        # status bar
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(240)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Open a video to begin.")

        # playback
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        # signals
        self.act_open.triggered.connect(self._open_dialog)
        self.act_analyze.triggered.connect(self._run_analysis)
        self.act_capture_fps.triggered.connect(self._set_capture_fps)
        self.act_calibrate.triggered.connect(self._start_calibration)
        self.act_export_csv.triggered.connect(self._export_csv)
        self.act_export_video.triggered.connect(self._export_video)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        # Apply saved defaults before connecting handlers to avoid churn.
        li = self.level_combo.findData(self.settings.athlete_level)
        if li >= 0:
            self.level_combo.setCurrentIndex(li)
        mi = self.model_combo.findData(self.settings.model_tier)
        if mi >= 0:
            self.model_combo.setCurrentIndex(mi)
        self.level_combo.currentIndexChanged.connect(self._on_level_changed)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.timeline.frame_requested.connect(self.seek)
        self.plot_panel.frame_requested.connect(self.seek)
        self.metrics_panel.frame_requested.connect(self.seek)
        self.form_panel.frame_requested.connect(self.seek)
        self.rep_card.frame_requested.connect(self.seek)
        self.compare_panel.frame_requested.connect(self.seek)
        self.step_charts.frame_requested.connect(self.seek)
        self.keyframe_strip.frame_requested.connect(self.seek)
        self.timeline.phase_zoom_requested.connect(self.plot_panel.zoom_to_frames)
        self.transport.step.connect(lambda d: self.seek(self.current_frame + d))
        self.transport.play_toggled.connect(self._on_play_toggled)
        self.transport.speed_changed.connect(lambda _s: self._update_timer_interval())
        self.video.point_picked.connect(self._on_point_picked)

        self._update_actions()

    # --- video loading -----------------------------------------------------

    def _open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video", "", "Videos (*.mp4 *.mov *.avi *.mkv);;All files (*)")
        if path:
            self.open_video(path)

    def open_video(self, path: str) -> None:
        self._video_path = path
        session = AnalysisSession.load(path, 30.0)
        # A cached analysis carries its own rotation; otherwise start unrotated
        # and let the assessment suggest one.
        rotation = session.rotation if session else 0
        try:
            source = VideoSource(path, rotation=rotation)
        except IOError as exc:
            QMessageBox.critical(self, "Open video", str(exc))
            return
        if self.source:
            self.source.close()
        self.source = source
        self.timeline.set_frame_count(source.frame_count)
        self.session = session or AnalysisSession(video_path=path, fps=source.fps)
        self.session.rotation = rotation
        idx = self.mode_combo.findData(self.session.mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        # Athlete level follows the user's current selection (global preference);
        # re-grade the loaded pose against it if it differs from the sidecar.
        level = self.level_combo.currentData()
        if self.session.athlete_level != level:
            self.session.athlete_level = level
            self.session.recompute()
        self.current_frame = 0
        self._refresh_analysis_views()
        self.seek(0)
        loaded = " (previous analysis loaded)" if self.session.has_pose else ""
        self.statusBar().showMessage(
            f"{Path(path).name}: {source.frame_count} frames @ {source.fps:.2f} fps{loaded}")
        self._start_assessment(path)
        self._update_actions()

    # --- suitability assessment --------------------------------------------

    def _start_assessment(self, path: str) -> None:
        if self._assessor is not None:
            return
        self.suitability_panel.set_busy()
        self._assessor = AssessmentWorker(path, self)
        self._assessor.finished_ok.connect(self._on_assessment_done)
        self._assessor.failed.connect(self._on_assessment_failed)
        self._assessor.start()

    def _on_assessment_done(self, assessment) -> None:
        self._finish_assessor()
        self.suitability_panel.show_assessment(assessment)
        # Auto-apply a suggested rotation for display — only if this clip has no
        # analysis yet (a cached result already fixed its own orientation).
        if (assessment.rotation_suggestion and self.source
                and not (self.session and self.session.has_pose)
                and self.source.rotation != assessment.rotation_suggestion):
            self._set_source_rotation(assessment.rotation_suggestion)

    def _on_assessment_failed(self, message: str) -> None:
        self._finish_assessor()
        self.statusBar().showMessage("Video assessment failed (analysis still works).")

    def _finish_assessor(self) -> None:
        if self._assessor:
            self._assessor.wait()
            self._assessor.deleteLater()
            self._assessor = None

    def _set_source_rotation(self, degrees: int) -> None:
        if not self.source:
            return
        self.source.close()
        self.source = VideoSource(self._video_path, rotation=degrees)
        if self.session:
            self.session.rotation = degrees
        self.timeline.set_frame_count(self.source.frame_count)
        self.seek(min(self.current_frame, self.source.frame_count - 1))
        self.statusBar().showMessage(f"Rotated {degrees}° for correct orientation.")

    # --- frame display -----------------------------------------------------

    def seek(self, frame: int) -> None:
        if not self.source:
            return
        frame = max(0, min(frame, self.source.frame_count - 1))
        self.current_frame = frame
        image = self.source.get_frame(frame)
        if image is None:
            return
        image = image.copy()
        s = self.session
        if s is not None and s.keypoints is not None and frame < len(s.keypoints):
            kpts = s.keypoints[frame]
            draw_pose(image, kpts)
            angles_now = {k: float(s.angles[k][frame])
                          for k in OVERLAY_ANGLES if k in s.angles}
            draw_angle_labels(image, kpts, angles_now)
            speed = s.velocities.get("run_speed")
            if speed is not None and frame < len(speed) and np.isfinite(speed[frame]):
                draw_info_text(image, f"speed {speed[frame]:.2f} {s.velocity_unit}",
                               row=1)
        self.video.set_frame(image)
        self.timeline.set_current(frame)
        self.plot_panel.set_current_frame(frame)
        self.transport.set_position(frame, self.source.frame_count,
                                    self.source.frame_to_time(frame))

    # --- playback ----------------------------------------------------------

    def _update_timer_interval(self) -> None:
        if self.source:
            self._timer.setInterval(
                max(5, round(1000 / (self.source.fps * self.transport.speed()))))

    def _on_play_toggled(self, playing: bool) -> None:
        if playing and self.source:
            self._update_timer_interval()
            self._timer.start()
        else:
            self._timer.stop()

    def _on_tick(self) -> None:
        if not self.source:
            return
        if self.current_frame >= self.source.frame_count - 1:
            self._timer.stop()
            self.transport.set_playing(False)
            return
        self.seek(self.current_frame + 1)

    # --- analysis ----------------------------------------------------------

    def _current_preprocessor(self) -> Preprocessor:
        sel = self.suitability_panel.selected_transforms()
        return Preprocessor(
            rotation=self.source.rotation if self.source else 0,
            reframe=sel.get("reframe", False),
            enhance=sel.get("enhance", False),
            deinterlace=sel.get("deinterlace", False))

    def _run_analysis(self) -> None:
        if not self.source or self._worker is not None:
            return
        tier = self.model_combo.currentData()
        pre = self._current_preprocessor()
        self._worker = AnalysisWorker(self.source.path, tier, pre, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_analysis_done)
        self._worker.failed.connect(self._on_analysis_failed)
        self.progress.setValue(0)
        self.progress.show()
        applied = pre.applied()
        tnote = f" · preprocessing: {', '.join(applied)}" if applied else ""
        note = " (Accurate: larger model, slower)" if tier == "Accurate" else ""
        if pre.needs_detection_pass():
            note += " · reframe adds a detection pass"
        self.statusBar().showMessage(
            f"Running pose analysis with {tier} model…{note}{tnote} "
            "(first run downloads the model)")
        self._pending_transforms = applied
        self.act_analyze.setEnabled(False)
        self._worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)

    def _on_analysis_done(self, keypoints: np.ndarray) -> None:
        self._finish_worker()
        if not self.session:
            return
        self.session.keypoints_raw = keypoints
        self.session.model_tier = self.model_combo.currentData()
        self.session.athlete_level = self.level_combo.currentData()
        self.session.rotation = self.source.rotation if self.source else 0
        self.session.transforms = getattr(self, "_pending_transforms", [])
        # Container metadata can overstate the frame count (seen with OGV);
        # the analysis pass decoded the whole file, so its length is the truth.
        if self.source and len(keypoints) < self.source.frame_count:
            self.source.frame_count = len(keypoints)
            self.timeline.set_frame_count(len(keypoints))
        self.session.recompute()
        try:
            self.session.save()
        except OSError as exc:
            self.statusBar().showMessage(f"Analysis done (couldn't save sidecar: {exc})")
        self._refresh_analysis_views()
        self.seek(self.current_frame)
        self.statusBar().showMessage("Pose analysis complete.")

    def _on_analysis_failed(self, message: str) -> None:
        self._finish_worker()
        QMessageBox.critical(self, "Pose analysis failed", message)
        self.statusBar().showMessage("Pose analysis failed.")

    def _finish_worker(self) -> None:
        if self._worker:
            self._worker.wait()
            self._worker.deleteLater()
            self._worker = None
        self.progress.hide()
        self._update_actions()

    def _refresh_analysis_views(self) -> None:
        s = self.session
        if not s or not s.has_pose:
            self.timeline.set_markers([])
            self.timeline.set_phases([])
            self.plot_panel.set_data({}, {}, self.source.fps if self.source else 30.0)
            self.metrics_panel.show_sprint(None)
            self.form_panel.show_findings([])
            self.plot_panel.set_phases([])
            self.rep_card.clear("Run pose analysis to see this rep's summary.")
            self.step_charts.set_steps([], "BH")
            self.keyframe_strip.set_keyframes([], lambda _f: None)
            self.compare_panel.set_comparisons([])
            self._update_actions()
            return
        self.plot_panel.set_data(s.angles, s.velocities, s.fps, s.velocity_unit)
        markers: list[tuple[int, QColor, str]] = []
        keyframes: list[tuple[int, str]] = []
        if s.mode == "sprint":
            for ev in s.gait_events:
                color = MARKER_COLORS[(ev.kind, ev.side)]
                markers.append((ev.frame, color, f"{ev.side} {ev.kind}"))
            has_steps = bool(s.sprint_metrics and s.sprint_metrics.steps)
            buckets = (bucket_sprint_steps(s.keypoints, s.sprint_metrics,
                                           s.velocities, s.fps)
                      if has_steps else {})
            self.metrics_panel.show_sprint(s.sprint_metrics, buckets, s.athlete_level)
            self.form_panel.show_findings(s.sprint_form, buckets, s.athlete_level)
            self.rep_card.show_sprint(s.sprint_metrics, s.sprint_form,
                                      s.quality, s.model_tier,
                                      radar=s.sprint_radar)
            spans = segment_phases(s.velocities.get("run_speed"), s.fps)
            self.plot_panel.set_phases(
                spans, targets=plot_target_bands(s.athlete_level))
            self.timeline.set_phases(spans)
            steps = s.sprint_metrics.steps if s.sprint_metrics else []
            self.step_charts.set_steps(steps, s.sprint_metrics.length_unit
                                       if s.sprint_metrics else "BH",
                                       buckets, s.athlete_level)
            for step in steps[:14]:
                knee = step.knee_angle_at_strike
                knee_txt = f" · knee {knee:.0f}°" if np.isfinite(knee) else ""
                keyframes.append((step.strike_frame,
                                  f"{step.side} strike f{step.strike_frame}{knee_txt}"))
            if has_steps:
                comparisons = build_comparisons(s.sprint_form, buckets, s.athlete_level)
                self.compare_panel.set_comparisons(comparisons)
            else:
                self.compare_panel.set_comparisons([])
        else:
            self.metrics_panel.show_jump(s.jump_metrics)
            self.form_panel.show_findings(s.jump_form)
            self.rep_card.show_jump(s.jump_metrics, s.jump_form,
                                    s.quality, s.model_tier)
            self.step_charts.set_steps([], "BH")
            self.compare_panel.set_comparisons(
                [], "Compare isn't available for jumps yet — it needs the "
                    "repeated per-phase steps a sprint has.")
            jump_spans: list[tuple[int, int, str]] = []
            if s.jump_phases:
                jp = s.jump_phases
                T = len(s.keypoints)
                markers = [
                    (jp.lowest_hip_frame, MARKER_COLORS["cm_bottom"], "CM bottom"),
                    (jp.takeoff_frame, MARKER_COLORS["takeoff"], "takeoff"),
                    (jp.landing_frame, MARKER_COLORS["landing"], "landing"),
                ]
                candidates = [
                    (max(0, jp.takeoff_frame - round(1.0 * s.fps)),
                     jp.lowest_hip_frame, "countermovement"),
                    (jp.lowest_hip_frame, jp.takeoff_frame, "drive up"),
                    (jp.takeoff_frame, jp.landing_frame, "flight"),
                    (jp.landing_frame,
                     min(T - 1, jp.landing_frame + round(0.5 * s.fps)), "landing"),
                ]
                jump_spans = [(a, b, n) for a, b, n in candidates if b > a]
                mid_flight = (jp.takeoff_frame + jp.landing_frame) // 2
                keyframes = [
                    (jp.lowest_hip_frame, f"CM bottom f{jp.lowest_hip_frame}"),
                    (jp.takeoff_frame, f"takeoff f{jp.takeoff_frame}"),
                    (mid_flight, f"peak f{mid_flight}"),
                    (jp.landing_frame, f"landing f{jp.landing_frame}"),
                ]
            self.plot_panel.set_phases(jump_spans)
            self.timeline.set_phases(jump_spans)
        self.keyframe_strip.set_keyframes(keyframes, self._render_keyframe)
        self.timeline.set_markers(markers)
        self._update_actions()

    def _render_keyframe(self, frame: int) -> np.ndarray | None:
        if not self.source:
            return None
        image = self.source.get_frame(frame)
        if image is None:
            return None
        image = image.copy()
        s = self.session
        if s is not None and s.keypoints is not None and frame < len(s.keypoints):
            draw_pose(image, s.keypoints[frame])
        return image

    def _render_keyframe_range(self, center_frame: int, half_window_frames: int,
                               max_frames: int = 16) -> list[np.ndarray]:
        """Every rendered frame in [center - half_window, center + half_window],
        subsampled to at most `max_frames` — a replay clip is meant to be a
        short, slow-motion loop, not a full re-decode of a high-fps clip's
        worth of frames every time a comparison card is built."""
        if not self.source:
            return []
        lo = max(0, center_frame - half_window_frames)
        hi = min(self.source.frame_count - 1, center_frame + half_window_frames)
        if hi < lo:
            return []
        frames = list(range(lo, hi + 1))
        if len(frames) > max_frames:
            idx = np.linspace(0, len(frames) - 1, max_frames)
            picked = [frames[round(i)] for i in idx]
            seen: set[int] = set()
            frames = [f for f in picked if not (f in seen or seen.add(f))]
        return [img for f in frames if (img := self._render_keyframe(f)) is not None]

    def _render_replay(self, center_frame: int) -> list[np.ndarray]:
        """A ~0.5 s-either-side replay clip around `center_frame` — the
        real-footage half of every ComparePanel/FormPanel visual comparison.
        Bundled here (rather than passing fps around) since only
        main_window knows the clip's fps."""
        fps = self.source.fps if self.source else 30.0
        return self._render_keyframe_range(center_frame, round(0.5 * fps))

    def _on_mode_changed(self) -> None:
        if self.session:
            self.session.mode = self.mode_combo.currentData()
            self._refresh_analysis_views()

    def _on_level_changed(self) -> None:
        level = self.level_combo.currentData()
        self.settings.athlete_level = level
        self.settings.save()
        # Grading is cheap — re-run it against the new level without touching pose.
        if self.session and self.session.has_pose:
            self.session.athlete_level = level
            self.session.recompute()
            try:
                self.session.save()
            except OSError:
                pass
            self._refresh_analysis_views()
            self.seek(self.current_frame)
            self.statusBar().showMessage(f"Re-graded for {level} level.")

    def _on_model_changed(self) -> None:
        self.settings.model_tier = self.model_combo.currentData()
        self.settings.save()
        if self.session and self.session.has_pose \
                and self.session.model_tier != self.settings.model_tier:
            self.statusBar().showMessage(
                f"Model set to {self.settings.model_tier}. Re-run pose analysis "
                "to apply it to this clip.")

    def _set_capture_fps(self) -> None:
        """Override the frame rate used for timing metrics — required for
        slow-motion footage where the container fps is not the capture fps
        (e.g. 240 fps phone video saved as a 30 fps file)."""
        if not self.session:
            return
        fps, ok = QInputDialog.getDouble(
            self, "Capture FPS",
            "Actual capture frame rate of the footage\n"
            "(differs from playback fps for slow-motion video):",
            self.session.fps, 1.0, 10000.0, 2)
        if not ok:
            return
        self.session.fps = fps
        self.session.recompute()
        if self.session.has_pose:
            self.session.save()
        self._refresh_analysis_views()
        self.statusBar().showMessage(f"Timing metrics now use {fps:g} fps.")

    # --- calibration -------------------------------------------------------

    def _start_calibration(self) -> None:
        if not self.source:
            return
        self._pick_points = []
        self.video.start_picking()
        self.statusBar().showMessage(
            "Calibration: click both ends of an object of known length in the video.")

    def _on_point_picked(self, x: float, y: float) -> None:
        self._pick_points.append((x, y))
        if len(self._pick_points) < 2:
            return
        p1, p2 = self._pick_points
        self.video.stop_picking()
        length, ok = QInputDialog.getDouble(
            self, "Calibration", "Real length of the drawn line (meters):",
            1.0, 0.01, 100.0, 3)
        if ok and self.session:
            self.session.calibration = Calibration.from_line(p1, p2, length)
            self.session.recompute()
            if self.session.has_pose:
                self.session.save()
            self._refresh_analysis_views()
            self.statusBar().showMessage(
                f"Calibrated: {self.session.calibration.meters_per_pixel * 1000:.2f} mm/px")

    # --- export ------------------------------------------------------------

    def _export_csv(self) -> None:
        s = self.session
        if not s or not s.has_pose:
            return
        base = str(Path(s.video_path).with_suffix(""))
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", base + "_frames.csv", "CSV (*.csv)")
        if not path:
            return
        export_frames_csv(s, path)
        metrics_path = path.replace("_frames.csv", "").replace(".csv", "") + "_metrics.csv"
        export_metrics_csv(s, metrics_path)
        self.statusBar().showMessage(f"Exported {Path(path).name} and {Path(metrics_path).name}")

    def _export_video(self) -> None:
        s = self.session
        if not s or not s.has_pose:
            return
        base = str(Path(s.video_path).with_suffix(""))
        path, _ = QFileDialog.getSaveFileName(
            self, "Export annotated video", base + "_annotated.mp4", "MP4 (*.mp4)")
        if not path:
            return
        self.progress.setValue(0)
        self.progress.show()

        def cb(done: int, total: int) -> None:
            self.progress.setMaximum(max(1, total))
            self.progress.setValue(done)
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()

        try:
            export_annotated_video(s, path, progress_cb=cb)
            self.statusBar().showMessage(f"Exported {Path(path).name}")
        except IOError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
        finally:
            self.progress.hide()

    # --- misc ----------------------------------------------------------------

    def _update_actions(self) -> None:
        has_video = self.source is not None
        has_pose = bool(self.session and self.session.has_pose)
        self.act_analyze.setEnabled(has_video and self._worker is None)
        self.act_capture_fps.setEnabled(has_video)
        self.act_calibrate.setEnabled(has_video)
        self.act_export_csv.setEnabled(has_pose)
        self.act_export_video.setEnabled(has_pose)

    def closeEvent(self, event) -> None:
        if self._worker:
            self._worker.cancel()
            self._worker.wait(5000)
        if self._assessor:
            self._assessor.wait(5000)
        if self.source:
            self.source.close()
        event.accept()
