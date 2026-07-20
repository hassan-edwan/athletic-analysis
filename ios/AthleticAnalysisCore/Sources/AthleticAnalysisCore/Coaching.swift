// Rules-based sprint form analysis (port of coaching.py, sprint path).
// Grades measured mechanics against literature reference ranges, tiered by
// athlete level and sprint phase. Range values and sources are transcribed
// verbatim from the Python core — that file remains the reference.

import Foundation

public enum AthleteLevel: String, CaseIterable, Sendable {
    case developmental, trained, elite
}

public enum SprintPhase: String, Sendable {
    case drive
    case acceleration
    case maxVelocity = "max velocity"
    case deceleration
}

public enum Severity: String, Sendable, Comparable {
    case major, minor, good

    var order: Int {
        switch self {
        case .major: return 0
        case .minor: return 1
        case .good: return 2
        }
    }

    public static func < (l: Severity, r: Severity) -> Bool { l.order < r.order }
}

public struct FormFinding: Sendable {
    public var phase: String
    public var metric: String
    public var value: Double
    public var valueText: String
    public var targetText: String
    public var severity: Severity
    public var cue: String
    public var frame: Int
    public var source: String
    public var confidence: MetricConfidence?
    public var key: String       // "trunk" | "contact_ms" | "knee_strike" | "thigh" | "cadence" | "overstride"
    public var deviation: String // "low" | "high" | ""
}

public struct Check: Sendable {
    public var metric: String
    public var unit: String  // "deg" | "ms" | "spm" | "BH" | "m" | custom
    public var lo: Double
    public var hi: Double
    public var tol: Double
    public var cueLow: String
    public var cueHigh: String
    public var source: String
    public var goodNote: String = "In the optimal range."
}

public enum Coaching {
    // MARK: - Formatting

    static func fmtValue(_ value: Double, unit: String) -> String {
        guard value.isFinite else { return "–" }
        switch unit {
        case "ms": return String(format: "%.0f ms", value * 1000)
        case "deg": return String(format: "%.0f°", value)
        case "spm": return String(format: "%.0f steps/min", value)
        default: return String(format: "%.2f %@", value, unit)
        }
    }

    static func fmtRange(_ check: Check) -> String {
        switch check.unit {
        case "ms": return String(format: "%.0f–%.0f ms", check.lo * 1000, check.hi * 1000)
        case "deg": return String(format: "%.0f–%.0f°", check.lo, check.hi)
        case "spm": return String(format: "%.0f–%.0f steps/min", check.lo, check.hi)
        default: return String(format: "%.2f–%.2f %@", check.lo, check.hi, check.unit)
        }
    }

    // MARK: - Evaluation

    static func evaluate(_ check: Check, value: Double, phase: String,
                         frame: Int, confidence: MetricConfidence? = nil,
                         key: String = "") -> FormFinding? {
        guard value.isFinite else { return nil }
        let severity: Severity
        let cue: String
        let deviation: String
        if check.lo <= value && value <= check.hi {
            (severity, cue, deviation) = (.good, check.goodNote, "")
        } else {
            let dist = value < check.lo ? check.lo - value : value - check.hi
            severity = dist <= check.tol ? .minor : .major
            deviation = value < check.lo ? "low" : "high"
            cue = deviation == "low" ? check.cueLow : check.cueHigh
        }
        return FormFinding(phase: phase, metric: check.metric, value: value,
                           valueText: fmtValue(value, unit: check.unit),
                           targetText: fmtRange(check), severity: severity,
                           cue: cue, frame: frame, source: check.source,
                           confidence: confidence, key: key, deviation: deviation)
    }

    // MARK: - Level-tiered reference numbers (verbatim from coaching.py)

    typealias Band = (lo: Double, hi: Double, tol: Double)

    static let contactS: [SprintPhase: [AthleteLevel: Band]] = [
        .drive: [.developmental: (0.16, 0.26, 0.06), .trained: (0.13, 0.22, 0.06),
                 .elite: (0.11, 0.18, 0.05)],
        .acceleration: [.developmental: (0.14, 0.22, 0.05),
                        .trained: (0.11, 0.18, 0.05), .elite: (0.10, 0.15, 0.04)],
        .maxVelocity: [.developmental: (0.11, 0.18, 0.05),
                       .trained: (0.095, 0.15, 0.04), .elite: (0.085, 0.12, 0.03)],
    ]
    static let cadenceBand: [AthleteLevel: Band] = [
        .developmental: (200, 300, 40), .trained: (220, 320, 35),
        .elite: (250, 340, 30),
    ]
    static let thighBand: [AthleteLevel: Band] = [
        .developmental: (45, 95, 15), .trained: (55, 100, 15),
        .elite: (65, 105, 12),
    ]
    static let kneeTD: [SprintPhase: [AthleteLevel: Band]] = [
        .drive: [.developmental: (120, 165, 12), .trained: (125, 165, 12),
                 .elite: (128, 165, 10)],
        .acceleration: [.developmental: (128, 168, 12),
                        .trained: (130, 168, 12), .elite: (135, 168, 10)],
        .maxVelocity: [.developmental: (135, 165, 12),
                       .trained: (140, 168, 10), .elite: (145, 170, 10)],
    ]

    static let contactSrc = "GCT: elite ~0.09 s at Vmax (Sides 2018; Nagahara)"
    static let trunkSrc = "trunk ~45° block exit → vertical at Vmax (auptimo; World Athletics)"
    static let cadSrc = "cadence rises with level; ~4–5 Hz elite (Nagahara)"
    static let fsSrc = "front-side mechanics / knee lift (Mann sprint model)"
    static let kneeSrc = "stiffer, more extended touchdown leg with level (Mann)"

    // MARK: - Check tables

    /// Phase -> [(metric key, Check)] for the given athlete level.
    public static func sprintChecks(_ level: AthleteLevel) -> [SprintPhase: [(String, Check)]] {
        func contact(_ phase: SprintPhase) -> Check {
            let b = contactS[phase]![level]!
            return Check(metric: "Ground contact time", unit: "ms",
                         lo: b.lo, hi: b.hi, tol: b.tol,
                         cueLow: "Contacts unusually short — verify capture FPS is "
                               + "set correctly and that you're finishing each push.",
                         cueHigh: "Long ground contacts — build stiffness and faster "
                                + "force production (bounds, sled/wall drills).",
                         source: contactSrc)
        }

        func kneeTd(_ phase: SprintPhase) -> Check {
            let b = kneeTD[phase]![level]!
            return Check(metric: "Knee angle at touchdown", unit: "deg",
                         lo: b.lo, hi: b.hi, tol: b.tol,
                         cueLow: "Knee collapsing at touchdown — cue a stiffer, "
                               + "spring-like leg on contact; build eccentric strength.",
                         cueHigh: "Landing on a near-locked leg — a slight knee bend "
                                + "absorbs and returns energy; land closer to the hips.",
                         source: kneeSrc)
        }

        let trunkDrive = Check(metric: "Trunk lean", unit: "deg", lo: 20, hi: 50, tol: 12,
            cueLow: "Too upright for the drive phase — stay low out of the start and "
                  + "push the ground back; let the body rise gradually.",
            cueHigh: "Leaning past ~50° — likely overreaching and losing balance.",
            source: trunkSrc)
        let trunkAccel = Check(metric: "Trunk lean", unit: "deg", lo: 8, hi: 35, tol: 10,
            cueLow: "Popping up too early — hold a gradual rise through the transition.",
            cueHigh: "Still fully crouched — allow the trunk to rise as speed builds.",
            source: trunkSrc)
        let trunkMaxv = Check(metric: "Trunk lean", unit: "deg", lo: -2, hi: 10, tol: 8,
            cueLow: "Leaning backward at top speed — run tall, eyes level.",
            cueHigh: "Excessive forward lean at max velocity kills front-side "
                   + "mechanics — run tall with hips under the shoulders.",
            source: trunkSrc)

        func overstride(_ lo: Double, _ hi: Double) -> Check {
            Check(metric: "Touchdown distance ahead of hip", unit: "BH",
                  lo: lo, hi: hi, tol: 0.06,
                  cueLow: "Foot landing well behind the hips — check tracking.",
                  cueHigh: "Overstriding — the foot planting ahead of the hips brakes "
                         + "you; cue 'step down and back', not reaching out.",
                  source: fsSrc)
        }

        let tb = thighBand[level]!
        let thigh = Check(metric: "Front-side knee lift (swing thigh)", unit: "deg",
                          lo: tb.lo, hi: tb.hi, tol: tb.tol,
                          cueLow: "Low knee lift — poor front-side mechanics; cue 'knees up, step "
                                + "over the opposite knee'.",
                          cueHigh: "Thigh beyond expected range — check tracking.",
                          source: fsSrc)
        let cb = cadenceBand[level]!
        let cadence = Check(metric: "Cadence", unit: "spm",
                            lo: cb.lo, hi: cb.hi, tol: cb.tol,
                            cueLow: "Low step rate — quicker, punchier steps; avoid overstriding.",
                            cueHigh: "Very high cadence — possibly cutting strides short; finish each "
                                   + "push.", source: cadSrc)

        return [
            .drive: [("trunk", trunkDrive), ("contact_ms", contact(.drive)),
                     ("knee_strike", kneeTd(.drive)),
                     ("overstride", overstride(-0.12, 0.08))],
            .acceleration: [("trunk", trunkAccel),
                            ("contact_ms", contact(.acceleration)),
                            ("knee_strike", kneeTd(.acceleration)),
                            ("overstride", overstride(-0.08, 0.12))],
            .maxVelocity: [("trunk", trunkMaxv),
                           ("contact_ms", contact(.maxVelocity)),
                           ("cadence", cadence),
                           ("knee_strike", kneeTd(.maxVelocity)),
                           ("thigh", thigh), ("overstride", overstride(-0.02, 0.14))],
        ]
    }

    // MARK: - Phase segmentation

    /// Per-frame phase segmentation for display: contiguous
    /// (startFrame, endFrame, phase) spans.
    public static func segmentPhases(_ runSpeed: [Double]?, fps: Double)
        -> [(Int, Int, SprintPhase)] {
        guard let rs = runSpeed else { return [] }
        let finiteCount = rs.filter { $0.isFinite }.count
        guard finiteCount >= 5 else { return [] }
        let vmax = NaNMath.nanMax(rs)
        guard vmax.isFinite, vmax > 0 else { return [] }
        var peak = 0
        var best = -Double.infinity
        for (i, v) in rs.enumerated() where v.isFinite && v > best {
            best = v
            peak = i
        }

        var labels = [SprintPhase?](repeating: nil, count: rs.count)
        var prev: SprintPhase? = nil
        for (i, v) in rs.enumerated() {
            guard v.isFinite else {
                labels[i] = prev
                continue
            }
            let r = v / vmax
            let cur: SprintPhase
            if r >= 0.93 {
                cur = .maxVelocity
            } else if i <= peak {
                cur = r < 0.70 ? .drive : .acceleration
            } else {
                cur = .deceleration
            }
            labels[i] = cur
            prev = cur
        }
        guard let first = labels.compactMap({ $0 }).first else { return [] }
        for i in 0..<labels.count {
            if labels[i] != nil { break }
            labels[i] = first
        }

        var spans: [(Int, Int, SprintPhase)] = []
        for (i, label) in labels.enumerated() {
            let c = label!
            if let last = spans.last, last.2 == c {
                spans[spans.count - 1].1 = i
            } else {
                spans.append((i, i, c))
            }
        }
        // Absorb blips shorter than ~0.15 s into the preceding span.
        let minLen = max(3, Int((0.15 * fps).rounded(.toNearestOrEven)))
        var merged: [(Int, Int, SprintPhase)] = []
        for sp in spans {
            if !merged.isEmpty && sp.1 - sp.0 + 1 < minLen {
                merged[merged.count - 1].1 = sp.1
            } else if let last = merged.last, last.2 == sp.2 {
                merged[merged.count - 1].1 = sp.1
            } else {
                merged.append(sp)
            }
        }
        return merged
    }

    static func phaseOfStep(strikeFrame: Int, runSpeed: [Double]?,
                            trunkLean: Double) -> SprintPhase {
        var ratio = Double.nan
        if let rs = runSpeed, strikeFrame >= 0, strikeFrame < rs.count {
            let vmax = NaNMath.nanMax(rs)
            if vmax.isFinite && vmax > 0 && rs[strikeFrame].isFinite {
                ratio = rs[strikeFrame] / vmax
            }
        }
        if ratio.isFinite {
            if ratio >= 0.93 { return .maxVelocity }
            if ratio < 0.70 { return .drive }
            return .acceleration
        }
        // No usable speed: fall back to posture.
        if trunkLean.isFinite && trunkLean > 25 { return .drive }
        return .maxVelocity
    }

    /// Horizontal distance (body-heights) the striking ankle lands ahead of
    /// the hip; positive = ahead of the body (braking).
    static func overstride(_ kpts: PoseSequence, strikeFrame: Int,
                           side: GaitEvent.Side, direction: Double,
                           bodyHPx: Double) -> Double {
        guard bodyHPx.isFinite, bodyHPx > 0,
              strikeFrame >= 0, strikeFrame < kpts.count else { return .nan }
        let ankle = kpts[strikeFrame][side == .left ? .lAnkle : .rAnkle]
        let hip = kpts[strikeFrame][.hipCenter]
        guard ankle.conf >= 0.3, hip.conf >= 0.3 else { return .nan }
        return (ankle.x - hip.x) * direction / bodyHPx
    }

    // MARK: - Phase bucketing (shared by form grading and radar scoring)

    public struct PhaseBucket {
        public var values: [String: [Double]] = [:]
        public var strikeFrames: [Int] = []
        public var contactFrames: [Double] = []  // contactTimeS * fps
        public var stepFrames: [Double] = []     // stepTimeS * fps
    }

    public static func medianOf(_ values: [Double]) -> Double {
        NaNMath.nanMedian(values)
    }

    public static func bucketSprintSteps(_ kpts: PoseSequence,
                                         sprint: SprintMetrics,
                                         runSpeed: [Double]?,
                                         fps: Double) -> [SprintPhase: PhaseBucket] {
        let direction = Angles.travelDirection(kpts)
        let bodyH = Angles.estimateBodyHeightPx(kpts)
        var buckets: [SprintPhase: PhaseBucket] = [:]
        for step in sprint.steps {
            let phase = phaseOfStep(strikeFrame: step.strikeFrame,
                                    runSpeed: runSpeed,
                                    trunkLean: step.trunkLeanAtStrike)
            var b = buckets[phase] ?? PhaseBucket()
            b.strikeFrames.append(step.strikeFrame)
            b.values["trunk", default: []].append(step.trunkLeanAtStrike)
            b.values["contact_ms", default: []].append(step.contactTimeS)
            b.values["knee_strike", default: []].append(step.kneeAngleAtStrike)
            b.values["thigh", default: []].append(step.swingThighAngle)
            let cadence = step.stepTimeS.isFinite && step.stepTimeS > 0
                ? 60.0 / step.stepTimeS : Double.nan
            b.values["cadence", default: []].append(cadence)
            b.values["overstride", default: []].append(
                overstride(kpts, strikeFrame: step.strikeFrame, side: step.side,
                           direction: direction, bodyHPx: bodyH))
            if step.contactTimeS.isFinite { b.contactFrames.append(step.contactTimeS * fps) }
            if step.stepTimeS.isFinite { b.stepFrames.append(step.stepTimeS * fps) }
            buckets[phase] = b
        }
        return buckets
    }

    // Joints whose tracking quality bounds each metric's confidence.
    static let metricJoints: [String: [KP]] = [
        "trunk": [.lShoulder, .rShoulder, .lHip, .rHip],
        "contact_ms": [.lHeel, .rHeel, .lBigToe, .rBigToe],
        "knee_strike": [.lHip, .rHip, .lKnee, .rKnee, .lAnkle, .rAnkle],
        "thigh": [.lHip, .rHip, .lKnee, .rKnee],
        "overstride": [.hipCenter, .lAnkle, .rAnkle],
        "cadence": [.hipCenter],
    ]

    // MARK: - Sprint form analysis

    public static func analyzeSprintForm(_ kpts: PoseSequence,
                                         sprint: SprintMetrics?,
                                         runSpeed: [Double]?,
                                         fps: Double,
                                         level: AthleteLevel = .trained) -> [FormFinding] {
        guard let sprint, !sprint.steps.isEmpty else { return [] }
        let checksByPhase = sprintChecks(level)
        let buckets = bucketSprintSteps(kpts, sprint: sprint,
                                        runSpeed: runSpeed, fps: fps)

        func confFor(_ key: String, _ bucket: PhaseBucket) -> MetricConfidence {
            var detection: Double? = nil
            if let joints = metricJoints[key] {
                detection = Confidence.detectionFactor(kpts, joints: joints,
                                                      frames: bucket.strikeFrames)
            }
            var spanned: Double? = nil
            if key == "contact_ms" {
                spanned = bucket.contactFrames.isEmpty ? 0.0
                    : NaNMath.nanMedian(bucket.contactFrames)
            } else if key == "cadence" {
                spanned = bucket.stepFrames.isEmpty ? 0.0
                    : NaNMath.nanMedian(bucket.stepFrames)
            }
            return Confidence.metricConfidence(detection: detection,
                                               framesSpanned: spanned,
                                               nSamples: bucket.strikeFrames.count)
        }

        var findings: [FormFinding] = []
        for phase in [SprintPhase.drive, .acceleration, .maxVelocity] {
            guard let bucket = buckets[phase] else { continue }
            let repFrame = bucket.strikeFrames[0]
            for (key, check) in checksByPhase[phase]! {
                if let finding = evaluate(check,
                                          value: medianOf(bucket.values[key] ?? []),
                                          phase: phase.rawValue, frame: repFrame,
                                          confidence: confFor(key, bucket),
                                          key: key) {
                    findings.append(finding)
                }
            }
        }
        return findings.enumerated()
            .sorted { ($0.element.severity.order, $0.element.frame, $0.offset)
                    < ($1.element.severity.order, $1.element.frame, $1.offset) }
            .map(\.element)
    }

    public static func summarize(_ findings: [FormFinding]) -> String {
        guard !findings.isEmpty else {
            return "No form checks available — run analysis first."
        }
        let good = findings.filter { $0.severity == .good }.count
        let minor = findings.filter { $0.severity == .minor }.count
        let major = findings.filter { $0.severity == .major }.count
        return "\(good)/\(findings.count) checks in optimal range · "
             + "\(minor) minor · \(major) major"
    }
}
