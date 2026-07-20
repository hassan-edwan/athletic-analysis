// Honest, heuristic confidence scoring for measured metrics
// (port of confidence.py). Deliberately coarse: three buckets, named limiter.

import Foundation

public struct MetricConfidence: Sendable, Equatable {
    public enum Level: String, Sendable { case high = "High", medium = "Medium", low = "Low" }

    public var score: Double
    public var level: Level
    public var limiter: String  // "" when High/unlimited
}

public struct ClipQuality: Sendable {
    public var detectionRate: Double
    public var fps: Double
    public var fpsAdequate: Bool
    public var calibrated: Bool
    public var level: MetricConfidence.Level
    public var notes: [String]
}

public enum Confidence {
    // Frames an event needs before timing is trustworthy.
    static let targetEventFrames = 7.0
    static let fullStepSample = 4.0

    static let fDetect = "joint tracking"
    static let fTime = "frame rate"
    static let fSample = "few steps"
    static let fCalib = "not calibrated"

    static func level(_ score: Double) -> MetricConfidence.Level {
        if score >= 0.75 { return .high }
        if score >= 0.5 { return .medium }
        return .low
    }

    /// Overall = product of factors; the smallest factor is the named limiter.
    static func combine(_ factors: [(name: String, value: Double)]) -> MetricConfidence {
        guard !factors.isEmpty else {
            return MetricConfidence(score: 1.0, level: .high, limiter: "")
        }
        let score = factors.reduce(1.0) { $0 * $1.value }
        let worst = factors.min { $0.value < $1.value }!
        let limiter = worst.value >= 0.75 ? "" : worst.name
        return MetricConfidence(score: score, level: level(score), limiter: limiter)
    }

    /// Mean keypoint confidence of `joints` over `frames` (0..1).
    static func detectionFactor(_ kpts: PoseSequence, joints: [KP],
                                frames: [Int]) -> Double {
        let valid = frames.filter { $0 >= 0 && $0 < kpts.count }
        guard !joints.isEmpty, !valid.isEmpty else { return 1.0 }
        var sum = 0.0
        var count = 0
        for f in valid {
            for j in joints {
                let c = kpts[f][j].conf
                if c.isFinite {
                    sum += c
                    count += 1
                }
            }
        }
        guard count > 0 else { return 1.0 }
        return min(1.0, max(0.0, sum / Double(count)))
    }

    /// How well the frame rate resolves an event of this many frames.
    static func temporalFactor(_ framesSpanned: Double) -> Double {
        guard framesSpanned.isFinite, framesSpanned > 0 else { return 0.4 }
        return min(1.0, max(0.25, framesSpanned / targetEventFrames))
    }

    /// Trust grows with the number of contributing reps/steps.
    static func sampleFactor(_ n: Int) -> Double {
        guard n > 0 else { return 0.4 }
        return min(1.0, max(0.4, Double(n) / fullStepSample))
    }

    /// Assemble a metric's confidence from whichever signals apply.
    public static func metricConfidence(detection: Double? = nil,
                                        framesSpanned: Double? = nil,
                                        nSamples: Int? = nil,
                                        uncalibratedDistance: Bool = false) -> MetricConfidence {
        var factors: [(String, Double)] = []
        if let detection { factors.append((fDetect, detection)) }
        if let framesSpanned { factors.append((fTime, temporalFactor(framesSpanned))) }
        if let nSamples { factors.append((fSample, sampleFactor(nSamples))) }
        let result = combine(factors)
        // Uncalibrated distance is relative, not wrong: cap at Medium.
        if uncalibratedDistance && result.level == .high {
            return MetricConfidence(score: min(result.score, 0.74),
                                    level: .medium, limiter: fCalib)
        }
        return result
    }

    /// Overall analysis-quality badge for the whole clip.
    public static func clipQuality(_ kpts: PoseSequence?, fps: Double,
                                   calibrated: Bool) -> ClipQuality {
        guard let kpts, !kpts.isEmpty else {
            return ClipQuality(detectionRate: 0, fps: fps, fpsAdequate: false,
                               calibrated: calibrated, level: .low,
                               notes: ["No pose data."])
        }
        let tracked = kpts.filter { $0[.hipCenter].conf >= Angles.minConf }.count
        let detectionRate = Double(tracked) / Double(kpts.count)
        let fpsAdequate = fps >= 60.0
        var notes = [String(format: "%.0f%% frames tracked", detectionRate * 100)]
        if !fpsAdequate {
            notes.append(String(format: "%.0f fps limits contact/flight timing (60+ recommended)", fps))
        }
        if !calibrated {
            notes.append("uncalibrated — distances/speeds in body-heights")
        }
        let score = min(1.0, max(0.0, detectionRate)) * (fpsAdequate ? 1.0 : 0.6)
        return ClipQuality(detectionRate: detectionRate, fps: fps,
                           fpsAdequate: fpsAdequate, calibrated: calibrated,
                           level: level(score), notes: notes)
    }
}
