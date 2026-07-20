# Athletic Analysis

Kinovea-inspired desktop app for **sprint and jump form analysis** with automatic
AI pose estimation (RTMPose, 26 keypoints including heels/toes).

Open a video, run pose analysis once, then scrub frame-by-frame with a skeleton
and live joint angles overlaid — while the app automatically detects foot
strikes, toe-offs, takeoff and landing, and derives form metrics.

## Features

- **Automatic pose overlay** — skeleton + per-joint angles (knee, hip, ankle,
  elbow, trunk lean) on every frame; zero-lag Butterworth filtering.
- **Per-frame velocity** — running speed (0.4 s-averaged horizontal velocity,
  shown on the video overlay and plotted over the whole run) plus raw hip
  horizontal/vertical velocity curves, all exported to CSV. Units are m/s when
  calibrated, body-heights/s otherwise.
- **Sprint mode** — per-step cadence, ground-contact time, flight time, step
  length, per-step speed, average and top speed, trunk lean, knee angle at
  touchdown, swing-leg (front-side) thigh angle.
- **Jump mode** — jump height from flight time (`g·t²/8`), takeoff velocity
  (`g·t/2`), hip-rise cross-check, countermovement depth, takeoff angles,
  landing knee flexion, knee/ankle separation ratio (frontal-view valgus proxy).
- **Form analysis with coaching cues** — the sprint is segmented into phases
  (drive / acceleration / max velocity) from the speed profile, and each
  phase's mechanics are graded against reference ranges: trunk lean, ground
  contact time, cadence, knee angle at touchdown, front-side knee lift, and
  overstriding. Jumps are graded on countermovement depth, takeoff extension,
  landing stiffness and knee valgus. Findings are ranked by severity with a
  cue for what to fix; clicking one jumps to the relevant frame.
- **Athlete-level tiering** — a Developmental / Trained / Elite selector shifts
  every reference range (e.g. elite max-velocity ground contact ~90 ms vs
  developmental ~150 ms). Each range carries its literature source (shown in
  tooltips and CSV). Changing level re-grades instantly — no re-analysis.
- **Confidence indicators** — every hero number and form finding carries a
  High / Medium / Low badge naming its dominant limiter (joint tracking, frame
  rate, calibration, or too few steps), plus a clip-level "Analysis quality"
  summary. Timing metrics are honestly downgraded on low-fps footage (a 90 ms
  contact at 30 fps is only ~3 frames); low-confidence findings are tagged
  "verify manually" and never headline the Rep Card over confident ones.
- **Pose model tiers** — Fast / Balanced / Accurate (RTMPose lightweight /
  balanced / performance). Accurate is more precise but downloads a larger
  model once and runs slower on CPU. The analysis sidecar records which model
  produced it, so you can re-run a clip on Accurate when a result looks off.
- **Video suitability assessment** — when you open a video, a fast pre-flight
  check (a detector sample, a few seconds) grades it Good / Fair / Poor and
  itemizes problems (low fps, small/off-center athlete, multiple people or
  split-view, poor lighting, motion blur, interlacing, wrong orientation)
  *before* the expensive pose pass.
- **Rescue transforms** — for fixable issues the app can preprocess the video:
  **auto-reframe** (track + crop + upscale the athlete — the biggest win for
  small/distant athletes and multi-panel clips), **contrast enhance** (CLAHE +
  gamma), **deinterlace**, and **auto-rotate**. Recommended fixes are
  pre-checked and applied on the next analysis; keypoints are mapped back to the
  full frame so overlays and metrics stay in display coordinates. Applied
  transforms are recorded in the sidecar and CSV. No fabricated frames — low fps
  is flagged, never faked by interpolation.
- **Rep Card** — the first thing you see after analysis: hero numbers (top
  speed, cadence, contact time / jump height, takeoff velocity), a form score,
  and the top issues to fix, each clickable to jump to the proving frame.
- **Per-step bar charts** — contact time, step length and speed per step,
  colored by leg, so asymmetry and step-to-step trends are visible instantly.
- **Key-frame strip** — a filmstrip of every touchdown (or CM-bottom / takeoff /
  peak / landing) with the pose drawn, for posture comparison across steps.
- **Kinovea-style video control** — frame stepping, slow motion (0.1×–1×),
  phase-tinted timeline with event markers (double-click a phase to zoom the
  graph to it), angle/velocity plots with named preset views (Speed, Ground
  contact, Posture), phase-colored backgrounds, optimal-range bands, and a
  hover crosshair with per-curve readouts.
- **Calibration** — click two points of known real-world length to get metric
  distances; uncalibrated lengths are reported in body-heights (BH).
- **Export** — per-frame kinematics CSV, metrics CSV, annotated MP4.
- Analysis results are cached in a `<video>.analysis.json` sidecar, so pose
  estimation runs only once per video.

## Install & run

```powershell
python -m venv .venv
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\athlete-analysis            # or: athlete-analysis path\to\clip.mp4
```

The first "Run Pose Analysis" downloads the RTMPose ONNX models (~120 MB, one
time). Everything runs locally on CPU; no GPU required. Expect roughly
1–3 frames/s of analysis on CPU — a 10 s clip takes a few minutes, and the
result is cached so it only happens once per video.

## Workflow

1. **Open Video** — side view for sprinting; side or front view for jumps.
2. **Run Pose Analysis** — progress bar runs over every frame.
3. Pick **Mode** (Sprint / Jump); event markers appear on the timeline and
   metrics fill in the right panel. Click any metric row to jump to its frame.
4. **Slow-motion footage:** use **Capture FPS…** to enter the real recording
   frame rate (e.g. 240 for iPhone slo-mo saved as a 30 fps file) — all timing
   metrics depend on it.
5. Optionally **Calibrate** with a known length in the shot to get meters.
6. **Export** CSVs or an annotated video.

## Filming tips

- 2D analysis is view-dependent: film perpendicular to the motion, camera still.
- Higher frame rate = better contact/flight timing (60–240 fps recommended;
  at 30 fps, contact times are only ±33 ms).
- Keep the whole athlete in frame; other people are tolerated (tracking prefers
  the person nearest the athlete's last position), but avoid them crossing in
  front of the athlete.

## Development

```powershell
.venv\Scripts\python -m pytest       # unit tests (synthetic trajectories)
```

Architecture: `core/` is UI-free (video decoding, pose backends, filtering,
angles, event detection, metrics) and fully unit-tested; `ui/` is PySide6;
`export/` writes CSV/MP4. Pose models are swappable via
`core/pose/base.PoseBackend`.
