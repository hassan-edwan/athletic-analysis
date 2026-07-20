// Gait event detection from foot trajectories (port of events.py, sprint path).
// Image coordinates: y grows downward — ground is at high y.

import Foundation

public struct GaitEvent: Sendable, Equatable {
    public enum Side: String, Sendable { case left, right }
    public enum Kind: String, Sendable { case strike, toeoff }

    public var frame: Int
    public var side: Side
    public var kind: Kind

    public init(frame: Int, side: Side, kind: Kind) {
        self.frame = frame
        self.side = side
        self.kind = kind
    }
}

public enum GaitEvents {
    /// Vertical position of the foot: mean of big toe and heel.
    static func footYSignal(_ kpts: PoseSequence, side: GaitEvent.Side) -> [Double] {
        let toe: KP = side == .left ? .lBigToe : .rBigToe
        let heel: KP = side == .left ? .lHeel : .rHeel
        return kpts.map { ($0[toe].y + $0[heel].y) / 2.0 }
    }

    /// [start, end) index pairs of true runs.
    static func runs(_ mask: [Bool]) -> [(Int, Int)] {
        var out: [(Int, Int)] = []
        var start: Int? = nil
        for (i, v) in mask.enumerated() {
            if v && start == nil { start = i }
            if !v, let s = start {
                out.append((s, i))
                start = nil
            }
        }
        if let s = start { out.append((s, mask.count)) }
        return out
    }

    static func closeGaps(_ mask: [Bool], maxGap: Int) -> [Bool] {
        var out = mask
        for (s, e) in runs(mask.map { !$0 }) {
            if s == 0 || e == mask.count { continue }  // keep leading/trailing gaps
            if e - s <= maxGap {
                for i in s..<e { out[i] = true }
            }
        }
        return out
    }

    static func dropShortRuns(_ mask: [Bool], minLen: Int) -> [Bool] {
        var out = mask
        for (s, e) in runs(mask) where e - s < minLen {
            for i in s..<e { out[i] = false }
        }
        return out
    }

    /// Boolean mask of frames where this foot is on the ground: near the
    /// ground level AND moving slowly (amplitude-scaled thresholds).
    public static func contactMask(_ footY: [Double], fps: Double,
                                   groundBand: Double = 0.15,
                                   speedFactor: Double = 2.5,
                                   minContactS: Double = 0.04,
                                   mergeGapS: Double = 0.03) -> [Bool] {
        let n = footY.count
        guard n >= 3 else { return [Bool](repeating: false, count: n) }
        let lo = NaNMath.nanPercentile(footY, 5)
        let hi = NaNMath.nanPercentile(footY, 95)
        let amp = hi - lo
        guard amp > 1e-9 else { return [Bool](repeating: true, count: n) }
        let ground = NaNMath.nanPercentile(footY, 90)
        let vy = NaNMath.gradient(footY).map { $0 * fps }
        var mask = [Bool](repeating: false, count: n)
        for i in 0..<n {
            let nearGround = footY[i] > ground - groundBand * amp
            let slow = abs(vy[i]) < speedFactor * amp
            mask[i] = nearGround && slow
        }
        // .toNearestOrEven matches Python's banker's-rounding round().
        mask = closeGaps(mask, maxGap: max(1, Int((mergeGapS * fps).rounded(.toNearestOrEven))))
        mask = dropShortRuns(mask, minLen: max(2, Int((minContactS * fps).rounded(.toNearestOrEven))))
        return mask
    }

    /// Foot strikes and toe-offs for both feet, sorted by frame.
    public static func detectGaitEvents(_ kpts: PoseSequence, fps: Double) -> [GaitEvent] {
        var events: [GaitEvent] = []
        for side in [GaitEvent.Side.left, .right] {
            let mask = contactMask(footYSignal(kpts, side: side), fps: fps)
            for (s, e) in runs(mask) {
                if s > 0 {
                    events.append(GaitEvent(frame: s, side: side, kind: .strike))
                }
                if e < mask.count {
                    events.append(GaitEvent(frame: e - 1, side: side, kind: .toeoff))
                }
            }
        }
        // Stable sort by frame preserves the left-then-right insertion order,
        // matching Python's list.sort stability.
        return events.enumerated()
            .sorted { ($0.element.frame, $0.offset) < ($1.element.frame, $1.offset) }
            .map(\.element)
    }
}
