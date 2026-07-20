// Observable app state: video import → pose pass → SprintAnalysis results.
// Mirrors the desktop AnalysisSession model: raw keypoints are the source of
// truth (persisted as a JSON sidecar in Documents); everything else recomputes
// instantly, so the Level picker re-grades without re-running pose.

import AthleticAnalysisCore
import AVFoundation
import Foundation
import Observation

@Observable
final class AnalysisStore {
    enum Phase: Equatable {
        case idle
        case importing
        case analyzing(progress: Double)  // 0…1 over frames
        case ready
        case failed(String)
    }

    var phase: Phase = .idle
    var videoURL: URL?
    var fps: Double = 30
    var level: AthleteLevel = .trained {
        didSet { recompute() }
    }

    // Raw pose track (source of truth) + derived analysis.
    private(set) var rawKeypoints: PoseSequence = []
    private(set) var analysis: SprintAnalysis?

    private let engine = PoseEngine()

    // MARK: - Import + analyze

    func importVideo(url: URL) async {
        phase = .importing
        videoURL = url
        do {
            let asset = AVURLAsset(url: url)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                phase = .failed("No video track found.")
                return
            }
            // Nominal frame rate = capture FPS (slow-mo clips report their
            // true high rate here — the wrong-FPS timing trap from desktop).
            fps = Double(try await track.load(.nominalFrameRate))
            let frameCount = try await estimatedFrameCount(asset: asset, track: track)

            phase = .analyzing(progress: 0)
            var poses: PoseSequence = []
            try await engine.run(asset: asset) { [weak self] pose, index in
                poses.append(pose)
                if index % 10 == 0 {
                    Task { @MainActor in
                        self?.phase = .analyzing(
                            progress: Double(index) / Double(max(1, frameCount)))
                    }
                }
            }
            rawKeypoints = poses
            try? saveSidecar()
            recompute()
            phase = .ready
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    func recompute() {
        guard !rawKeypoints.isEmpty else {
            analysis = nil
            return
        }
        analysis = SprintAnalysis.analyze(rawKeypoints: rawKeypoints,
                                          fps: fps, level: level)
    }

    private func estimatedFrameCount(asset: AVAsset, track: AVAssetTrack) async throws -> Int {
        let duration = try await asset.load(.duration).seconds
        return Int(duration * fps)
    }

    /// Back to the import screen for a new clip — "raw keypoints are the
    /// source of truth" applies to clearing state too, not just recomputing it.
    func reset() {
        phase = .idle
        videoURL = nil
        rawKeypoints = []
        analysis = nil
    }

    // MARK: - Sidecar persistence (raw keypoints only, like desktop)

    private var sidecarURL: URL? {
        guard let videoURL else { return nil }
        let docs = FileManager.default.urls(for: .documentDirectory,
                                            in: .userDomainMask)[0]
        return docs.appendingPathComponent(
            videoURL.deletingPathExtension().lastPathComponent + ".analysis.json")
    }

    func saveSidecar() throws {
        guard let url = sidecarURL else { return }
        let payload: [String: Any] = [
            "version": 1,
            "fps": fps,
            "level": level.rawValue,
            "keypoints_raw": rawKeypoints.map { pose in
                pose.points.map { [round($0.x * 100) / 100,
                                  round($0.y * 100) / 100,
                                  round($0.conf * 100) / 100] }
            },
        ]
        let data = try JSONSerialization.data(withJSONObject: payload)
        try data.write(to: url)
    }

    func loadSidecar(for videoURL: URL) -> Bool {
        self.videoURL = videoURL
        guard let url = sidecarURL,
              let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let raw = obj["keypoints_raw"] as? [[[Double]]] else {
            return false
        }
        fps = obj["fps"] as? Double ?? 30
        if let lv = obj["level"] as? String, let l = AthleteLevel(rawValue: lv) {
            level = l
        }
        rawKeypoints = raw.map { frame in
            Pose(points: frame.map { Keypoint(x: $0[0], y: $0[1], conf: $0[2]) })
        }
        recompute()
        phase = .ready
        return true
    }
}
