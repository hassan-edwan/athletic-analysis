// Halpe-26 keypoint layout and bone connections (port of pose/skeleton.py).
// Rendering lives in the app layer; the core only needs indices and names.

import Foundation

public enum KP: Int, CaseIterable, Sendable {
    case nose = 0, lEye, rEye, lEar, rEar
    case lShoulder, rShoulder, lElbow, rElbow, lWrist, rWrist
    case lHip, rHip, lKnee, rKnee, lAnkle, rAnkle
    case head, neck, hipCenter
    case lBigToe, rBigToe, lSmallToe, rSmallToe, lHeel, rHeel

    public static let count = 26

    public var name: String {
        Skeleton.names[rawValue]
    }
}

public enum Skeleton {
    public static let names: [String] = [
        "nose", "l_eye", "r_eye", "l_ear", "r_ear",
        "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
        "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
        "head", "neck", "hip_center",
        "l_big_toe", "r_big_toe", "l_small_toe", "r_small_toe", "l_heel", "r_heel",
    ]

    public static let bones: [(KP, KP)] = [
        (.head, .neck), (.nose, .head),
        (.neck, .lShoulder), (.neck, .rShoulder), (.neck, .hipCenter),
        (.lShoulder, .lElbow), (.lElbow, .lWrist),
        (.rShoulder, .rElbow), (.rElbow, .rWrist),
        (.hipCenter, .lHip), (.hipCenter, .rHip),
        (.lHip, .lKnee), (.lKnee, .lAnkle),
        (.rHip, .rKnee), (.rKnee, .rAnkle),
        (.lAnkle, .lHeel), (.lAnkle, .lBigToe), (.lBigToe, .lSmallToe),
        (.rAnkle, .rHeel), (.rAnkle, .rBigToe), (.rBigToe, .rSmallToe),
    ]
}

/// One keypoint: image-pixel position + model confidence.
public struct Keypoint: Sendable, Equatable {
    public var x: Double
    public var y: Double
    public var conf: Double

    public init(x: Double, y: Double, conf: Double) {
        self.x = x
        self.y = y
        self.conf = conf
    }
}

/// One frame of pose: exactly 26 keypoints, indexable by `KP`.
public struct Pose: Sendable {
    public var points: [Keypoint]

    public init(points: [Keypoint]) {
        precondition(points.count == KP.count)
        self.points = points
    }

    public subscript(_ kp: KP) -> Keypoint {
        get { points[kp.rawValue] }
        set { points[kp.rawValue] = newValue }
    }
}

/// A clip's pose track: (T, 26, 3) in the Python core.
public typealias PoseSequence = [Pose]

public extension PoseSequence {
    /// Column extraction, mirroring `kpts[:, KP[name], axis]`.
    func series(_ kp: KP, _ axis: KeyPath<Keypoint, Double>) -> [Double] {
        map { $0[kp][keyPath: axis] }
    }
}
