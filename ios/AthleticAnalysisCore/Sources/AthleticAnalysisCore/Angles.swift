// Joint angles from Halpe-26 keypoint trajectories (port of angles.py).
// All series are per-frame degrees with NaN where keypoints are low-confidence.

import Foundation

public enum Angles {
    public static let minConf = 0.3

    /// Interior angle at b (degrees) for a single frame.
    static func angle3pt(a: Keypoint, b: Keypoint, c: Keypoint) -> Double {
        let v1x = a.x - b.x, v1y = a.y - b.y
        let v2x = c.x - b.x, v2y = c.y - b.y
        let dot = v1x * v2x + v1y * v2y
        let norm = (v1x * v1x + v1y * v1y).squareRoot() * (v2x * v2x + v2y * v2y).squareRoot()
        let cosv = max(-1.0, min(1.0, dot / norm))  // NaN propagates
        return acos(cosv) * 180.0 / .pi
    }

    static func jointAngle(_ kpts: PoseSequence, _ a: KP, _ b: KP, _ c: KP) -> [Double] {
        kpts.map { pose in
            let pa = pose[a], pb = pose[b], pc = pose[c]
            if pa.conf < minConf || pb.conf < minConf || pc.conf < minConf {
                return .nan
            }
            return angle3pt(a: pa, b: pb, c: pc)
        }
    }

    /// Overall horizontal direction of motion: +1 rightward, -1 leftward.
    public static func travelDirection(_ kpts: PoseSequence) -> Double {
        var xs = [Double]()
        for pose in kpts where pose[.hipCenter].conf >= minConf {
            xs.append(pose[.hipCenter].x)
        }
        guard xs.count >= 2 else { return 1.0 }
        return xs[xs.count - 1] >= xs[0] ? 1.0 : -1.0
    }

    /// All angle time-series used by the app; kpts should be smoothed.
    public static func computeAngles(_ kpts: PoseSequence) -> [String: [Double]] {
        let direction = travelDirection(kpts)
        var out: [String: [Double]] = [
            "knee_l": jointAngle(kpts, .lHip, .lKnee, .lAnkle),
            "knee_r": jointAngle(kpts, .rHip, .rKnee, .rAnkle),
            "hip_l": jointAngle(kpts, .lShoulder, .lHip, .lKnee),
            "hip_r": jointAngle(kpts, .rShoulder, .rHip, .rKnee),
            "ankle_l": jointAngle(kpts, .lKnee, .lAnkle, .lBigToe),
            "ankle_r": jointAngle(kpts, .rKnee, .rAnkle, .rBigToe),
            "elbow_l": jointAngle(kpts, .lShoulder, .lElbow, .lWrist),
            "elbow_r": jointAngle(kpts, .rShoulder, .rElbow, .rWrist),
        ]

        // trunk_lean: mid-hip -> mid-shoulder vs image vertical, + = forward.
        out["trunk_lean"] = kpts.map { pose in
            let ls = pose[.lShoulder], rs = pose[.rShoulder]
            let lh = pose[.lHip], rh = pose[.rHip]
            let conf = min(min(ls.conf, rs.conf), min(lh.conf, rh.conf))
            if conf < minConf { return .nan }
            let topX = (ls.x + rs.x) / 2, topY = (ls.y + rs.y) / 2
            let botX = (lh.x + rh.x) / 2, botY = (lh.y + rh.y) / 2
            let dx = (topX - botX) * direction
            let dy = botY - topY  // positive when shoulders above hips
            return atan2(dx, dy) * 180.0 / .pi
        }

        // thigh vs vertical, + = knee ahead of the hip in the travel direction.
        for (name, hipKP, kneeKP) in [("thigh_l", KP.lHip, KP.lKnee),
                                      ("thigh_r", KP.rHip, KP.rKnee)] {
            out[name] = kpts.map { pose in
                let hip = pose[hipKP], knee = pose[kneeKP]
                if min(hip.conf, knee.conf) < minConf { return .nan }
                let dx = (knee.x - hip.x) * direction
                let dy = knee.y - hip.y  // positive: knee below hip
                return atan2(dx, dy) * 180.0 / .pi
            }
        }
        return out
    }

    /// Approximate stature in pixels from median segment lengths.
    public static func estimateBodyHeightPx(_ kpts: PoseSequence) -> Double {
        func seg(_ a: KP, _ b: KP) -> Double {
            var dists = [Double]()
            for pose in kpts {
                let pa = pose[a], pb = pose[b]
                guard pa.conf >= minConf && pb.conf >= minConf else { continue }
                dists.append((pow(pa.x - pb.x, 2) + pow(pa.y - pb.y, 2)).squareRoot())
            }
            guard dists.count >= 3 else { return .nan }
            return NaNMath.nanMedian(dists)
        }

        let shank = NaNMath.nanMean([seg(.lKnee, .lAnkle), seg(.rKnee, .rAnkle)])
        let thigh = NaNMath.nanMean([seg(.lHip, .lKnee), seg(.rHip, .rKnee)])
        let trunk = NaNMath.nanMean([seg(.lHip, .lShoulder), seg(.rHip, .rShoulder)])
        let head = seg(.neck, .head)
        if !shank.isFinite || !thigh.isFinite || !trunk.isFinite { return .nan }
        let headTerm = head.isFinite ? head * 1.3 : trunk * 0.45
        // Ankle-to-floor offset approximated as 5% of the sum.
        return (shank + thigh + trunk + headTerm) * 1.05
    }
}
