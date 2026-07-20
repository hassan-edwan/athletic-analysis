// Filmstrip of every foot-strike with the pose drawn on top, for side-by-side
// posture comparison across steps (port of the desktop ui/keyframe_strip.py).
// Thumbnails are decoded lazily and cached in @State as they arrive.

import AthleticAnalysisCore
import AVFoundation
import SwiftUI
import UIKit

struct KeyframeStripView: View {
    let videoURL: URL
    let analysis: SprintAnalysis
    let fps: Double
    var onSeek: (Int) -> Void = { _ in }

    @State private var thumbnails: [Int: UIImage] = [:]
    @State private var videoSize: CGSize = .zero

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionHeader(title: "Touchdowns", trailing: "\(analysis.metrics.steps.count) steps")
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(Array(analysis.metrics.steps.enumerated()), id: \.offset) { index, step in
                        cell(index: index, step: step)
                    }
                }
                .padding(.horizontal, 2)
                .padding(.vertical, 2)
            }
        }
        .task { await loadThumbnails() }
    }

    private var cellAspect: CGFloat {
        videoSize == .zero ? 9.0 / 16.0 : videoSize.height / videoSize.width
    }

    private func cell(index: Int, step: StepRecord) -> some View {
        Button {
            onSeek(step.strikeFrame)
        } label: {
            VStack(spacing: 4) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(Theme.surfaceRaised)
                    if let image = thumbnails[step.strikeFrame], videoSize != .zero {
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(videoSize.width / videoSize.height, contentMode: .fit)
                        SkeletonOverlay(pose: pose(at: step.strikeFrame),
                                        videoSize: videoSize, viewSize: .zero)
                    } else {
                        ProgressView().scaleEffect(0.7)
                    }
                }
                .frame(width: 92, height: 92 * cellAspect)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(Theme.leg(step.side), lineWidth: 2)
                )
                Text("#\(index + 1)")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(Theme.leg(step.side))
            }
        }
        .buttonStyle(.plain)
    }

    private func pose(at frame: Int) -> Pose? {
        guard frame >= 0, frame < analysis.keypoints.count else { return nil }
        return analysis.keypoints[frame]
    }

    private func loadThumbnails() async {
        let asset = AVURLAsset(url: videoURL)
        guard let track = try? await asset.loadTracks(withMediaType: .video).first,
              let natural = try? await track.load(.naturalSize) else { return }
        videoSize = natural

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 240, height: 240)

        for step in analysis.metrics.steps {
            guard thumbnails[step.strikeFrame] == nil else { continue }
            let time = CMTime(seconds: Double(step.strikeFrame) / fps, preferredTimescale: 600)
            if let result = try? await generator.image(at: time) {
                thumbnails[step.strikeFrame] = UIImage(cgImage: result.image)
            }
        }
    }
}

/// Hosts the filmstrip and the per-step trend charts together — both are
/// indexed by step, so they share one tab.
struct StepsView: View {
    let videoURL: URL
    let analysis: SprintAnalysis
    let fps: Double
    var onSeek: (Int) -> Void = { _ in }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                KeyframeStripView(videoURL: videoURL, analysis: analysis, fps: fps, onSeek: onSeek)
                StepChartsView(analysis: analysis, fps: fps)
            }
            .padding()
        }
        .background(Theme.background)
    }
}
