// Video player with skeleton overlay + a phase-tinted scrubber (the app's
// PhaseRibbon signature) carrying foot-strike markers and a live playhead.
// The overlay stays in sync during native playback too, not just scrubbing.

import AthleticAnalysisCore
import AVKit
import SwiftUI

struct PlayerView: View {
    let videoURL: URL
    let analysis: SprintAnalysis
    let fps: Double
    @Binding var currentFrame: Int

    @State private var player: AVPlayer?
    @State private var videoSize: CGSize = .zero
    @State private var timeObserverToken: Any?

    var body: some View {
        VStack(spacing: 14) {
            videoCard
            scrubber
        }
        .padding(.top, 8)
        .task { await setUpPlayer() }
        .onDisappear {
            if let token = timeObserverToken { player?.removeTimeObserver(token) }
        }
        .onChange(of: currentFrame) { _, frame in
            seekPlayer(toFrame: frame)
        }
    }

    // MARK: - Video + overlay

    private var videoCard: some View {
        ZStack(alignment: .top) {
            ZStack {
                VideoPlayer(player: player)
                GeometryReader { geo in
                    SkeletonOverlay(pose: pose(at: currentFrame),
                                    videoSize: videoSize, viewSize: geo.size)
                        .allowsHitTesting(false)
                }
            }
            .aspectRatio(videoSize == .zero ? 16.0 / 9.0
                         : videoSize.width / videoSize.height,
                         contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous))

            hud.padding(8)
        }
    }

    private var hud: some View {
        HStack {
            if let phase = currentPhase {
                Chip(text: phase.rawValue.capitalized, color: Theme.phase(phase), filled: true)
            }
            Spacer()
            Text(currentSpeedText)
                .font(.caption.monospacedDigit().weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(.black.opacity(0.45), in: Capsule())
        }
    }

    private func pose(at frame: Int) -> Pose? {
        guard frame >= 0, frame < analysis.keypoints.count else { return nil }
        return analysis.keypoints[frame]
    }

    // MARK: - Scrubber (PhaseRibbon + step ticks + playhead)

    private var totalFrames: Int { analysis.keypoints.count }

    private var phaseSpans: [(Int, Int, SprintPhase)] {
        Coaching.segmentPhases(analysis.velocities.runSpeed, fps: fps)
    }

    private var currentPhase: SprintPhase? {
        phaseSpans.first { $0.0 <= currentFrame && currentFrame <= $0.1 }?.2
    }

    private var currentSpeedText: String {
        let rs = analysis.velocities.runSpeed
        guard currentFrame >= 0, currentFrame < rs.count, rs[currentFrame].isFinite else {
            return "– \(analysis.velocities.unit)"
        }
        return "\(String(format: "%.2f", rs[currentFrame])) \(analysis.velocities.unit)"
    }

    private var frameLabel: String {
        "Frame \(currentFrame + 1) / \(max(1, totalFrames))"
    }

    private var scrubber: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(frameLabel)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                Spacer()
                PhaseLegend()
            }

            let total = max(1, totalFrames - 1)
            GeometryReader { geo in
                ZStack(alignment: .topLeading) {
                    PhaseRibbon(spans: phaseSpans, totalFrames: totalFrames, height: 16)
                    ForEach(Array(analysis.metrics.steps.enumerated()), id: \.offset) { _, step in
                        let x = geo.size.width * CGFloat(step.strikeFrame) / CGFloat(total)
                        Circle()
                            .fill(Theme.leg(step.side))
                            .frame(width: 8, height: 8)
                            .overlay(Circle().stroke(Theme.background, lineWidth: 1.5))
                            .position(x: x, y: 8)
                            .onTapGesture { currentFrame = step.strikeFrame }
                    }
                    let px = geo.size.width * CGFloat(currentFrame) / CGFloat(total)
                    RoundedRectangle(cornerRadius: 1)
                        .fill(Color.white)
                        .frame(width: 2, height: 20)
                        .position(x: px, y: 8)
                }
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0).onChanged { value in
                        let ratio = max(0, min(1, value.location.x / geo.size.width))
                        currentFrame = Int((ratio * CGFloat(total)).rounded())
                    }
                )
            }
            .frame(height: 20)
        }
        .padding(.horizontal)
    }

    // MARK: - Player lifecycle

    private func setUpPlayer() async {
        let p = AVPlayer(url: videoURL)
        player = p
        if let track = try? await AVURLAsset(url: videoURL)
            .loadTracks(withMediaType: .video).first,
           let size = try? await track.load(.naturalSize) {
            videoSize = size
        }
        let interval = CMTime(seconds: 1.0 / max(fps, 1), preferredTimescale: 600)
        timeObserverToken = p.addPeriodicTimeObserver(forInterval: interval, queue: .main) { time in
            let frame = Int((time.seconds * fps).rounded())
            if frame != currentFrame, frame >= 0, frame < totalFrames {
                currentFrame = frame
            }
        }
    }

    /// Only seek when the player has actually drifted from `frame` by more
    /// than half a frame — otherwise the periodic time observer above (which
    /// mirrors natural playback into `currentFrame`) fights its own seeks.
    private func seekPlayer(toFrame frame: Int) {
        guard let player else { return }
        let target = CMTime(seconds: Double(frame) / fps, preferredTimescale: 600)
        let deltaFrames = abs(target.seconds - player.currentTime().seconds) * fps
        guard deltaFrames > 0.5 else { return }
        player.seek(to: target, toleranceBefore: .zero, toleranceAfter: .zero)
    }
}

struct SkeletonOverlay: View {
    let pose: Pose?
    let videoSize: CGSize
    let viewSize: CGSize

    private let confThreshold = 0.3

    var body: some View {
        Canvas { ctx, size in
            guard let pose, videoSize.width > 0, videoSize.height > 0 else { return }
            // Aspect-fit mapping from video pixels to view points.
            let scale = min(size.width / videoSize.width,
                            size.height / videoSize.height)
            let offsetX = (size.width - videoSize.width * scale) / 2
            let offsetY = (size.height - videoSize.height * scale) / 2
            func map(_ kp: Keypoint) -> CGPoint {
                CGPoint(x: offsetX + kp.x * scale, y: offsetY + kp.y * scale)
            }
            func boneColor(_ a: KP, _ b: KP) -> Color {
                let names = (a.name, b.name)
                if names.0.hasPrefix("l_") || names.1.hasPrefix("l_") { return Theme.legLeft }
                if names.0.hasPrefix("r_") || names.1.hasPrefix("r_") { return Theme.legRight }
                return Theme.accent
            }
            for (a, b) in Skeleton.bones {
                let pa = pose[a]
                let pb = pose[b]
                guard pa.conf >= confThreshold, pb.conf >= confThreshold else {
                    continue
                }
                var path = Path()
                path.move(to: map(pa))
                path.addLine(to: map(pb))
                ctx.stroke(path, with: .color(boneColor(a, b)), lineWidth: 2.5)
            }
            for kp in KP.allCases {
                let p = pose[kp]
                guard p.conf >= confThreshold else { continue }
                let pt = map(p)
                let dot = Path(ellipseIn: CGRect(x: pt.x - 2.5, y: pt.y - 2.5,
                                                 width: 5, height: 5))
                ctx.fill(dot, with: .color(boneColor(kp, kp)))
            }
        }
    }
}
