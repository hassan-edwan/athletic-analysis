// Sprint metrics from gait events + angle series (port of metrics/sprint.py).
// Distances in meters when calibrated, else body-heights (BH).

import Foundation

public struct StepRecord: Sendable {
    public var side: GaitEvent.Side
    public var strikeFrame: Int
    public var toeoffFrame: Int?
    public var contactTimeS: Double
    public var flightTimeS: Double
    public var stepTimeS: Double
    public var stepLength: Double  // m or BH
    public var stepSpeed: Double   // m/s or BH/s
    public var kneeAngleAtStrike: Double
    public var swingThighAngle: Double  // opposite thigh vs vertical at strike
    public var trunkLeanAtStrike: Double
}

public struct SprintMetrics: Sendable {
    public var steps: [StepRecord] = []
    public var cadenceSpm: Double = .nan
    public var meanContactS: Double = .nan
    public var meanFlightS: Double = .nan
    public var meanStepLength: Double = .nan
    public var meanSpeed: Double = .nan
    public var maxSpeed: Double = .nan
    public var meanTrunkLeanDeg: Double = .nan
    public var lengthUnit: String = "BH"
    public var bodyHeightPx: Double = .nan

    public init() {}
}

public enum SprintMetricsComputer {
    static func at(_ series: [Double]?, _ frame: Int) -> Double {
        guard let series, frame >= 0, frame < series.count else { return .nan }
        return series[frame]
    }

    public static func compute(_ kpts: PoseSequence, angles: [String: [Double]],
                               events: [GaitEvent], fps: Double,
                               calib: Calibration? = nil) -> SprintMetrics {
        var m = SprintMetrics()
        let strikes = events.filter { $0.kind == .strike }
        guard !strikes.isEmpty else { return m }

        let bodyHPx = Angles.estimateBodyHeightPx(kpts)
        m.bodyHeightPx = bodyHPx
        if calib != nil { m.lengthUnit = "m" }

        func pxToLen(_ px: Double) -> Double {
            if let calib { return calib.toMeters(px) }
            if bodyHPx.isFinite && bodyHPx > 0 { return px / bodyHPx }
            return .nan
        }

        let hipX = kpts.series(.hipCenter, \.x)

        for (i, strike) in strikes.enumerated() {
            let sideKey = strike.side == .left ? "l" : "r"
            let other = sideKey == "l" ? "r" : "l"
            let toeoff = events.first {
                $0.kind == .toeoff && $0.side == strike.side && $0.frame > strike.frame
            }
            let nxt: GaitEvent? = i + 1 < strikes.count ? strikes[i + 1] : nil

            var contactS = Double.nan
            if let toeoff, nxt == nil || toeoff.frame <= nxt!.frame {
                contactS = Double(toeoff.frame - strike.frame) / fps
            }
            var flightS = Double.nan
            if let toeoff, let nxt, nxt.frame > toeoff.frame {
                flightS = Double(nxt.frame - toeoff.frame) / fps
            }
            var stepS = Double.nan
            var stepLen = Double.nan
            var stepSpeed = Double.nan
            if let nxt {
                stepS = Double(nxt.frame - strike.frame) / fps
                stepLen = pxToLen(abs(hipX[nxt.frame] - hipX[strike.frame]))
                if stepS > 0 { stepSpeed = stepLen / stepS }
            }

            m.steps.append(StepRecord(
                side: strike.side,
                strikeFrame: strike.frame,
                toeoffFrame: toeoff?.frame,
                contactTimeS: contactS,
                flightTimeS: flightS,
                stepTimeS: stepS,
                stepLength: stepLen,
                stepSpeed: stepSpeed,
                kneeAngleAtStrike: at(angles["knee_\(sideKey)"], strike.frame),
                swingThighAngle: at(angles["thigh_\(other)"], strike.frame),
                trunkLeanAtStrike: at(angles["trunk_lean"], strike.frame)))
        }

        let meanStepTime = NaNMath.nanMean(m.steps.map(\.stepTimeS))
        if meanStepTime.isFinite && meanStepTime > 0 {
            m.cadenceSpm = 60.0 / meanStepTime
        }
        let first = strikes[0].frame
        let last = strikes[strikes.count - 1].frame
        if last > first {
            m.meanSpeed = pxToLen(abs(hipX[last] - hipX[first]))
                / (Double(last - first) / fps)
            let vx = NaNMath.gradient(hipX).map { abs($0) * fps }
            let window = max(3, Int((0.4 * fps).rounded(.toNearestOrEven)))
            let smoothVx = NaNMath.rollingNanMean(vx, window: window)
            let slice = Array(smoothVx[first...last])
            let peak = NaNMath.nanMax(slice)
            if peak.isFinite { m.maxSpeed = pxToLen(peak) }
        }
        m.meanContactS = NaNMath.nanMean(m.steps.map(\.contactTimeS))
        m.meanFlightS = NaNMath.nanMean(m.steps.map(\.flightTimeS))
        m.meanStepLength = NaNMath.nanMean(m.steps.map(\.stepLength))
        m.meanTrunkLeanDeg = NaNMath.nanMean(m.steps.map(\.trunkLeanAtStrike))
        return m
    }
}
