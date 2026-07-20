// Sprint factor profile for the pentagon chart (port of radar.py).
// Scores are normalized against the same level-tiered bands the grader uses,
// via the shared phase bucketing, so the radar can never disagree with the
// findings table.

import Foundation

public enum RadarAxis: String, CaseIterable, Sendable {
    case stiffness = "Stiffness / contact"
    case frontSide = "Front-side mechanics"
    case posture = "Posture / trunk"
    case footPlacement = "Foot placement"
    case rhythm = "Rhythm"

    var metricKeys: [String] {
        switch self {
        case .stiffness: return ["contact_ms"]
        case .frontSide: return ["thigh"]
        case .posture: return ["trunk"]
        case .footPlacement: return ["overstride", "knee_strike"]
        case .rhythm: return ["cadence"]
        }
    }
}

public struct AxisScore: Sendable {
    public var name: String
    public var score: Double  // 0..100; NaN = no data
    public var detail: String
    public var nSteps: Int
}

public struct SprintRadar: Sendable {
    public var axes: [AxisScore] = []  // RadarAxis order
    public var overall: Double = .nan
    public var level: AthleteLevel = .trained
}

public enum Radar {
    /// Symmetry index at or beyond which the symmetry sub-score hits zero.
    static let symZero = 0.15

    /// Map a measured value to 0–100 against an optimal band: in band = 100,
    /// minor zone 100→60, major zone 60→0 (floor at 3 tolerances beyond).
    public static func bandScore(_ value: Double, lo: Double, hi: Double,
                                 tol: Double) -> Double {
        guard value.isFinite else { return .nan }
        let d = max(lo - value, value - hi, 0.0)
        if d == 0.0 { return 100.0 }
        if tol <= 0 { return 0.0 }
        if d <= tol { return 100.0 - 40.0 * (d / tol) }
        return max(0.0, 60.0 - 60.0 * ((d - tol) / (2.0 * tol)))
    }

    /// 0–100 left/right symmetry from per-side medians of step time and
    /// contact time; NaN when neither quantity has a finite value per side.
    public static func symmetryScore(_ steps: [StepRecord]) -> Double {
        var subs = [Double]()
        for attr in [\StepRecord.stepTimeS, \StepRecord.contactTimeS] {
            let left = NaNMath.nanMedian(steps.filter { $0.side == .left }
                .map { $0[keyPath: attr] })
            let right = NaNMath.nanMedian(steps.filter { $0.side == .right }
                .map { $0[keyPath: attr] })
            let mean = 0.5 * (left + right)
            guard left.isFinite, right.isFinite, mean > 0 else { continue }
            let si = abs(left - right) / mean
            subs.append(100.0 * min(1.0, max(0.0, 1.0 - si / symZero)))
        }
        return subs.isEmpty ? .nan : subs.reduce(0, +) / Double(subs.count)
    }

    public static func computeSprintRadar(_ kpts: PoseSequence,
                                          sprint: SprintMetrics?,
                                          runSpeed: [Double]?,
                                          fps: Double,
                                          level: AthleteLevel = .trained) -> SprintRadar? {
        guard let sprint, !sprint.steps.isEmpty else { return nil }
        let checksByPhase = Coaching.sprintChecks(level)
        let buckets = Coaching.bucketSprintSteps(kpts, sprint: sprint,
                                                 runSpeed: runSpeed, fps: fps)

        var radar = SprintRadar(level: level)
        for axis in RadarAxis.allCases {
            // (score, weight, detail) per contributing (phase, metric).
            var contributions: [(score: Double, weight: Double, detail: String)] = []
            // Iterate phases in a fixed order for deterministic detail strings.
            for phase in [SprintPhase.drive, .acceleration, .maxVelocity] {
                guard let bucket = buckets[phase],
                      let checks = checksByPhase[phase] else { continue }
                let checkMap = Dictionary(uniqueKeysWithValues: checks)
                for key in axis.metricKeys {
                    guard let check = checkMap[key] else { continue }
                    let value = Coaching.medianOf(bucket.values[key] ?? [])
                    let score = bandScore(value, lo: check.lo, hi: check.hi,
                                          tol: check.tol)
                    guard score.isFinite else { continue }
                    let detail = "\(check.metric.lowercased()) "
                        + "\(Coaching.fmtValue(value, unit: check.unit)) vs "
                        + "\(Coaching.fmtRange(check)) (\(phase.rawValue))"
                    contributions.append((score, Double(bucket.strikeFrames.count),
                                          detail))
                }
            }
            if axis == .rhythm {
                let sym = symmetryScore(sprint.steps)
                if sym.isFinite {
                    contributions.append(
                        (sym, Double(sprint.steps.count),
                         String(format: "L/R symmetry %.0f/100 over %d steps",
                                sym, sprint.steps.count)))
                }
            }
            let axisScore: AxisScore
            if !contributions.isEmpty {
                let totalWeight = contributions.reduce(0) { $0 + $1.weight }
                let weighted = contributions.reduce(0) { $0 + $1.score * $1.weight }
                let dominant = contributions.max { $0.weight < $1.weight }!.detail
                var seen = Set<String>()
                let details = ([dominant] + contributions.map(\.detail))
                    .filter { seen.insert($0).inserted }
                axisScore = AxisScore(name: axis.rawValue,
                                      score: weighted / totalWeight,
                                      detail: details.joined(separator: "; "),
                                      nSteps: Int(contributions.map(\.weight).max() ?? 0))
            } else {
                axisScore = AxisScore(name: axis.rawValue, score: .nan,
                                      detail: "no data in this clip", nSteps: 0)
            }
            radar.axes.append(axisScore)
        }

        let finite = radar.axes.map(\.score).filter { $0.isFinite }
        radar.overall = finite.isEmpty ? .nan
            : finite.reduce(0, +) / Double(finite.count)
        return radar
    }
}
