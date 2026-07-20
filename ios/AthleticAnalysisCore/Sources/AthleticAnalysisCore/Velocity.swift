// Per-frame velocity series from smoothed keypoints (port of velocity.py).
// Units: m/s when calibrated, otherwise body-heights per second (BH/s).

import Foundation

/// Pixel-to-meter scale from a user-calibrated reference length.
public struct Calibration: Sendable {
    public var metersPerPixel: Double

    public init(metersPerPixel: Double) {
        self.metersPerPixel = metersPerPixel
    }

    public func toMeters(_ px: Double) -> Double {
        px * metersPerPixel
    }
}

public struct Velocities: Sendable {
    public var hipVx: [Double]
    public var hipVy: [Double]  // up-positive
    public var hipSpeed: [Double]
    public var runSpeed: [Double]  // ~0.4 s-averaged |vx| — the coach's speed
    public var unit: String
}

public enum Velocity {
    public static func computeVelocities(_ kpts: PoseSequence, fps: Double,
                                         calib: Calibration? = nil) -> Velocities {
        let scale: Double
        let unit: String
        if let calib {
            scale = calib.metersPerPixel
            unit = "m/s"
        } else {
            let bodyH = Angles.estimateBodyHeightPx(kpts)
            if bodyH.isFinite && bodyH > 0 {
                scale = 1.0 / bodyH
                unit = "BH/s"
            } else {
                scale = .nan
                unit = "?/s"
            }
        }

        let T = kpts.count
        guard T >= 2 else {
            let empty = [Double](repeating: .nan, count: T)
            return Velocities(hipVx: empty, hipVy: empty, hipSpeed: empty,
                              runSpeed: empty, unit: unit)
        }
        let hipX = kpts.series(.hipCenter, \.x)
        let hipY = kpts.series(.hipCenter, \.y)
        let hipConf = kpts.series(.hipCenter, \.conf)
        var vx = NaNMath.gradient(hipX).map { $0 * fps * scale }
        var vy = NaNMath.gradient(hipY).map { -$0 * fps * scale }  // up-positive
        for t in 0..<T where hipConf[t] < Angles.minConf {
            vx[t] = .nan
            vy[t] = .nan
        }
        var run = NaNMath.rollingNanMean(vx.map { abs($0) },
                                         window: max(3, Int((0.4 * fps).rounded(.toNearestOrEven))))
        for t in 0..<T where hipConf[t] < Angles.minConf {
            run[t] = .nan
        }
        let speed = zip(vx, vy).map { (pow($0, 2) + pow($1, 2)).squareRoot() }
        return Velocities(hipVx: vx, hipVy: vy, hipSpeed: speed,
                          runSpeed: run, unit: unit)
    }
}
