# Athletic Analysis — iOS (Sprint MVP)

Native SwiftUI port of the desktop analyzer. See the plan in the repo root's
planning notes: on-device RTMPose (Halpe-26) via ONNX Runtime + a pure-Swift
analysis core, developed and proven **on Windows**; the Xcode app shell is
built in batched Mac sessions.

## Layout

- `AthleticAnalysisCore/` — SwiftPM package: the analysis math ported 1:1 from
  `athletic_analysis/core/` (sprint path only). No UIKit/CoreML — builds with
  the plain Swift toolchain on Windows/Linux/macOS.
- `AthleticAnalysisCore/Tests/.../Fixtures/*.json` — golden fixtures generated
  from the **Python core** (the reference implementation) by
  `tools/export_fixtures.py`.
- `AthleticAnalysis/` — app sources for the Xcode target (created on the Mac;
  these files compile only there). Pre-written on Windows:
  - `App/AthleticAnalysisApp.swift`, `App/Models/AnalysisStore.swift` —
    @Observable store: import → pose pass → `SprintAnalysis.analyze`, raw-
    keypoint sidecar persistence, instant Level re-grade, `reset()` for
    "New Clip".
  - `App/DesignSystem.swift` — the shared visual language: `Theme` (palette,
    typography), `Chip`, `.cardStyle()`, `SectionHeader`. Phase/severity/
    confidence colors are the *exact* RGB values the desktop app already uses
    (`ui/plot_panel.py PHASE_COLORS`, `ui/rep_card.py _CONF_COLOR`,
    `ui/radar_widget.py` score bands) so the two apps read as one product;
    only the brand accent (a cool "timing display" blue) is iOS-only, chosen
    to stay clear of every data color.
  - `App/PhaseRibbon.swift` — the app's signature element: a run's phase
    composition (drive/acceleration/max velocity/deceleration) as one tinted
    ribbon. The same component (and colors) draws the video scrubber
    background, the Rep Card's "shape of this rep" strip, and the mini strip
    above each per-step chart — a rep's phase structure reads the same way
    everywhere it appears.
  - `App/Views/` — `HomeView` (PhotosPicker import, styled progress/error
    states, 4-tab results shell with a "New Clip" reset), `PlayerView`
    (AVPlayer + Canvas skeleton overlay; scrubber is now a PhaseRibbon with
    foot-strike ticks, a drag-to-seek playhead, and a live phase/speed HUD;
    a periodic time observer keeps the skeleton in sync during native
    playback too, not just scrubbing), `RepCardView` (clip-quality card,
    phase-overview strip, 6-tile hero grid, pentagon, coaching banner, top
    issues as tappable cards), `RadarView` (Canvas pentagon, recolored via
    `Theme`), `FindingsView` (All/Issues segmented filter, phase + confidence
    chips, expandable cause/muscle/drill diagnostics — keyed by
    phase+metric so filtering never desyncs expand state), `StepChartsView`
    (**new** — Swift Charts bar charts of contact time / step length / step
    speed per step, colored by leg, phase-ribbon strip per chart; port of
    `ui/step_charts.py`), `KeyframeStripView` (**new** — AVAssetImageGenerator
    thumbnail filmstrip of every touchdown with the pose overlaid, for
    posture comparison across steps; port of `ui/keyframe_strip.py`),
    `StepsView` (**new** — hosts the filmstrip + charts as one tab).
  - `Pose/PoseEngine.swift` — AVAssetReader frames → vImage letterbox/warp →
    ONNX Runtime (Core ML EP) → PoseProcessing decode; det-every-5-frames
    tracking like the desktop backend.

## Windows workflow

```
# one-time: install the Swift toolchain (needs ~5 GB free disk)
winget install --id Swift.Toolchain

# regenerate goldens after any Python-core change
python tools/export_fixtures.py        # analysis-core fixtures
python tools/export_pose_fixtures.py   # pose pre/post-processing fixtures

# build + parity-test the Swift core (wrapper sets up MSVC + SDKROOT)
ios\swift-env.bat test
```

Status 2026-07-20: **all 13 parity tests pass** on Windows (Swift 6.3.3) —
9 analysis-core fixtures (incl. the full synthetic-gait pipeline) + 4 pose
pre/post-processing suites.

The parity tests assert the Swift port reproduces Python outputs: angles
±0.05°, event frames exact, step timing exact, findings (key/severity/
deviation/diagnosis) exact, radar scores ±(1e-6 module / 0.05 pipeline).

Module-to-module mapping (Python → Swift):

| Python (`athletic_analysis/core/`) | Swift (`Sources/AthleticAnalysisCore/`) |
|---|---|
| `pose/skeleton.py` | `Skeleton.swift` |
| `filtering.py` (+ scipy butter/filtfilt) | `Filtering.swift` |
| `angles.py` | `Angles.swift` |
| `velocity.py` | `Velocity.swift` |
| `events.py` (sprint) | `GaitEvents.swift` |
| `metrics/sprint.py` | `SprintMetrics.swift` |
| `coaching.py` (sprint) | `Coaching.swift` |
| `diagnostics.py` | `Diagnostics.swift` |
| `radar.py` | `Radar.swift` |
| `confidence.py` | `Confidence.swift` |
| `session.py` recompute (sprint) | `SprintAnalysis.swift` |
| rtmlib YOLOX/RTMPose pre+post, `rtmpose_backend.select_person`, `detector.select_tracked_box` | `PoseProcessing.swift` |

The scipy `butter`/`filtfilt`/`lfilter_zi` and numpy `gradient`/`interp`/
`nanpercentile`/`median_filter` ports were verified against scipy/numpy to
≤1e-9 (see the algorithm-validation script in the session scratchpad; the
JSON fixtures re-verify end to end on every `swift test`).

## Mac sessions (next)

1. Create the Xcode app project (`AthleticAnalysis/`), add the local
   `AthleticAnalysisCore` package and the `onnxruntime-objc` pod/SPM package
   with the Core ML execution provider.
2. Bundle the rtmlib ONNX models (person detector + RTMPose body26); port the
   pre/post-processing (letterbox+NMS, affine crop 288×384, SimCC decode) —
   written as package code so tensors recorded from the desktop app can test
   it off-device too.
3. SwiftUI MVP: PhotosPicker import (nominal frame rate = capture FPS),
   AVAssetReader analysis pass with progress, AVPlayer + Canvas overlay,
   Rep Card (hero tiles + pentagon Canvas + top issues), findings list with
   expandable diagnosis, level picker re-grading instantly.
4. **First compile pass of the redesigned UI** (2026-07-20 batch, written on
   Windows without a SwiftUI compiler — expect nits): `DesignSystem.swift`,
   `PhaseRibbon.swift`, and the reworked `HomeView`/`PlayerView`/`RepCardView`/
   `FindingsView`/`RadarView` plus the new `StepChartsView`/
   `KeyframeStripView`/`StepsView`. Things to specifically verify on-device:
   - `StepChartsView`'s Swift Charts `Chart(data) { … }` / `BarMark` usage.
   - `KeyframeStripView`'s `AVAssetImageGenerator.image(at:)` thumbnail loop —
     correctness and perf for a ~15–20 step clip (currently sequential, not
     parallelized).
   - `PlayerView`'s periodic time observer ↔ `currentFrame` binding feedback
     loop (seeks are gated on >0.5-frame drift so playback and scrubbing
     don't fight) — confirm no stutter during native playback.
   - SF Symbol names used (`shoeprints.fill`, `waveform.path.ecg`, etc.) — a
     wrong name renders blank, not a crash, but worth a visual pass.
