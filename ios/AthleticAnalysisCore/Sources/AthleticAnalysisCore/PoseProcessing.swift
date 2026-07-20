// Pose-model pre/post-processing (port of rtmlib's YOLOX + RTMPose paths and
// the app's person selection). Pure math only — the app layer does the actual
// pixel resampling (vImage/CoreGraphics) and ONNX Runtime inference; this file
// computes letterbox geometry, affine matrices, detection decoding/NMS, SimCC
// decoding, and coordinate mapping, so it is parity-testable on any platform.
//
// Faithful-quirk notes (kept identical to rtmlib, do not "fix"):
// - RTMPose normalization uses mean (123.675, 116.28, 103.53) applied to the
//   BGR image without channel swap.
// - YOLOX keeps detections with final score > nms_thr (0.45), not score_thr.
// - Tensor layout for both models: (1, 3, H, W) float32 from the BGR image.

import Foundation

// MARK: - Geometry primitives

public struct AffineMatrix: Sendable, Equatable {
    // Row-major 2x3: [a b c; d e f] mapping (x, y) -> (ax+by+c, dx+ey+f).
    public var m: (Double, Double, Double, Double, Double, Double)

    public func apply(x: Double, y: Double) -> (x: Double, y: Double) {
        (m.0 * x + m.1 * y + m.2, m.3 * x + m.4 * y + m.5)
    }

    public static func == (l: AffineMatrix, r: AffineMatrix) -> Bool {
        l.m == r.m
    }
}

/// cv2.getAffineTransform: the 2x3 matrix mapping 3 source points onto 3
/// destination points (solves two 3x3 linear systems).
public func affineTransform(src: [(Double, Double)],
                            dst: [(Double, Double)]) -> AffineMatrix {
    precondition(src.count == 3 && dst.count == 3)
    // Solve [x y 1] * [a d; b e; c f] = [dx dy] per destination coordinate.
    func solve3(_ rows: [[Double]], _ rhs: [Double]) -> [Double] {
        var a = rows
        var b = rhs
        for col in 0..<3 {
            var pivot = col
            for r in (col + 1)..<3 where abs(a[r][col]) > abs(a[pivot][col]) {
                pivot = r
            }
            a.swapAt(col, pivot)
            b.swapAt(col, pivot)
            for r in (col + 1)..<3 {
                let f = a[r][col] / a[col][col]
                if f == 0 { continue }
                for k in col..<3 { a[r][k] -= f * a[col][k] }
                b[r] -= f * b[col]
            }
        }
        var x = [Double](repeating: 0, count: 3)
        for r in stride(from: 2, through: 0, by: -1) {
            var s = b[r]
            for k in (r + 1)..<3 { s -= a[r][k] * x[k] }
            x[r] = s / a[r][r]
        }
        return x
    }
    let rows = src.map { [$0.0, $0.1, 1.0] }
    let xs = solve3(rows, dst.map(\.0))
    let ys = solve3(rows, dst.map(\.1))
    return AffineMatrix(m: (xs[0], xs[1], xs[2], ys[0], ys[1], ys[2]))
}

// MARK: - YOLOX person detection

public struct DetectionBox: Sendable {
    public var x1: Double
    public var y1: Double
    public var x2: Double
    public var y2: Double
    public var score: Double

    public var height: Double { y2 - y1 }
    public var center: (x: Double, y: Double) { ((x1 + x2) / 2, (y1 + y2) / 2) }
}

public enum YOLOXProcessing {
    public static let padValue: UInt8 = 114
    public static let nmsThreshold = 0.45
    public static let scoreThreshold = 0.7

    /// Letterbox geometry (the app resizes pixels; this defines the target):
    /// scale by `ratio`, paste at top-left of a `padValue`-gray canvas.
    public struct Letterbox: Sendable {
        public var ratio: Double
        public var resizedWidth: Int
        public var resizedHeight: Int
    }

    public static func letterbox(imageWidth: Int, imageHeight: Int,
                                 inputWidth: Int, inputHeight: Int) -> Letterbox {
        if imageWidth == inputWidth && imageHeight == inputHeight {
            return Letterbox(ratio: 1.0, resizedWidth: imageWidth,
                             resizedHeight: imageHeight)
        }
        let ratio = min(Double(inputHeight) / Double(imageHeight),
                        Double(inputWidth) / Double(imageWidth))
        return Letterbox(ratio: ratio,
                         resizedWidth: Int(Double(imageWidth) * ratio),
                         resizedHeight: Int(Double(imageHeight) * ratio))
    }

    /// Decode raw YOLOX output (N, 5 + classes) laid out flat, with the
    /// stride-grid offsets, into person boxes in original-image coordinates,
    /// then NMS. `values` is row-major (N, cols).
    public static func decode(values: [Double], cols: Int,
                              inputWidth: Int, inputHeight: Int,
                              ratio: Double,
                              nmsThr: Double = nmsThreshold,
                              scoreThr: Double = scoreThreshold) -> [DetectionBox] {
        let n = values.count / cols
        precondition(cols > 5, "expected raw (no-NMS) YOLOX output")
        // Grid centers and strides in row order: stride 8, 16, 32.
        var grids: [(gx: Double, gy: Double, stride: Double)] = []
        for stride in [8, 16, 32] {
            let h = inputHeight / stride
            let w = inputWidth / stride
            for y in 0..<h {
                for x in 0..<w {
                    grids.append((Double(x), Double(y), Double(stride)))
                }
            }
        }
        precondition(grids.count == n, "grid count \(grids.count) != rows \(n)")

        var boxes: [DetectionBox] = []
        var scoresPerBox: [Double] = []
        for i in 0..<n {
            let base = i * cols
            let g = grids[i]
            let cx = (values[base] + g.gx) * g.stride
            let cy = (values[base + 1] + g.gy) * g.stride
            let bw = exp(values[base + 2]) * g.stride
            let bh = exp(values[base + 3]) * g.stride
            // Person = class 0; score = objectness * class score.
            let score = values[base + 4] * values[base + 5]
            boxes.append(DetectionBox(x1: (cx - bw / 2) / ratio,
                                      y1: (cy - bh / 2) / ratio,
                                      x2: (cx + bw / 2) / ratio,
                                      y2: (cy + bh / 2) / ratio,
                                      score: score))
            scoresPerBox.append(score)
        }
        let valid = boxes.filter { $0.score > scoreThr }
        let kept = nms(valid, threshold: nmsThr)
        // rtmlib quirk: after NMS it additionally keeps score > nms_thr.
        return kept.filter { $0.score > nmsThr }
    }

    /// Single-class NMS (rtmlib's numpy implementation, +1 area convention).
    public static func nms(_ boxes: [DetectionBox], threshold: Double) -> [DetectionBox] {
        guard !boxes.isEmpty else { return [] }
        let order = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }
        var keep: [DetectionBox] = []
        var remaining = order
        while let i = remaining.first {
            let bi = boxes[i]
            keep.append(bi)
            let areaI = (bi.x2 - bi.x1 + 1) * (bi.y2 - bi.y1 + 1)
            remaining = remaining.dropFirst().filter { j in
                let bj = boxes[j]
                let xx1 = max(bi.x1, bj.x1)
                let yy1 = max(bi.y1, bj.y1)
                let xx2 = min(bi.x2, bj.x2)
                let yy2 = min(bi.y2, bj.y2)
                let w = max(0.0, xx2 - xx1 + 1)
                let h = max(0.0, yy2 - yy1 + 1)
                let inter = w * h
                let areaJ = (bj.x2 - bj.x1 + 1) * (bj.y2 - bj.y1 + 1)
                return inter / (areaI + areaJ - inter) <= threshold
            }
        }
        return keep
    }
}

// MARK: - RTMPose top-down crop + SimCC decode

public enum RTMPoseProcessing {
    /// BGR channel means/stds (rtmlib applies these without channel swap).
    public static let mean = (123.675, 116.28, 103.53)
    public static let std = (58.395, 57.12, 57.375)
    public static let simccSplitRatio = 2.0
    public static let bboxPadding = 1.25

    public struct CropSpec: Sendable {
        public var center: (x: Double, y: Double)
        public var scale: (w: Double, h: Double)  // aspect-fixed, padded
        public var warp: AffineMatrix             // image -> model input
    }

    /// bbox_xyxy2cs + aspect fix + get_warp_matrix (rot 0, no shift).
    public static func cropSpec(bbox: (x1: Double, y1: Double, x2: Double, y2: Double),
                                inputWidth: Int, inputHeight: Int) -> CropSpec {
        let center = ((bbox.x1 + bbox.x2) * 0.5, (bbox.y1 + bbox.y2) * 0.5)
        var scaleW = (bbox.x2 - bbox.x1) * bboxPadding
        var scaleH = (bbox.y2 - bbox.y1) * bboxPadding
        let aspect = Double(inputWidth) / Double(inputHeight)
        if scaleW > scaleH * aspect {
            scaleH = scaleW / aspect
        } else {
            scaleW = scaleH * aspect
        }
        // get_warp_matrix with rot=0, shift=(0,0):
        // src: center; center + (0, -srcW/2); 3rd point (90° CCW).
        // dst: input center; + (0, -dstW/2); 3rd point.
        func third(_ a: (Double, Double), _ b: (Double, Double)) -> (Double, Double) {
            let dir = (a.0 - b.0, a.1 - b.1)
            return (b.0 - dir.1, b.1 + dir.0)
        }
        let srcP0 = center
        let srcP1 = (center.0, center.1 - scaleW * 0.5)
        let srcP2 = third(srcP0, srcP1)
        let dstW = Double(inputWidth)
        let dstH = Double(inputHeight)
        let dstP0 = (dstW * 0.5, dstH * 0.5)
        let dstP1 = (dstW * 0.5, dstH * 0.5 - dstW * 0.5)
        let dstP2 = third(dstP0, dstP1)
        let warp = affineTransform(src: [srcP0, srcP1, srcP2],
                                   dst: [dstP0, dstP1, dstP2])
        return CropSpec(center: center, scale: (scaleW, scaleH), warp: warp)
    }

    /// get_simcc_maximum + coordinate mapping back to image space.
    /// simccX: (K, Wx) row-major; simccY: (K, Wy) row-major.
    public static func decodeSimCC(simccX: [Double], widthX: Int,
                                   simccY: [Double], widthY: Int,
                                   keypointCount: Int,
                                   crop: CropSpec,
                                   inputWidth: Int, inputHeight: Int) -> [Keypoint] {
        var out: [Keypoint] = []
        out.reserveCapacity(keypointCount)
        for k in 0..<keypointCount {
            var maxX = -Double.infinity
            var argX = 0
            for i in 0..<widthX {
                let v = simccX[k * widthX + i]
                if v > maxX {
                    maxX = v
                    argX = i
                }
            }
            var maxY = -Double.infinity
            var argY = 0
            for i in 0..<widthY {
                let v = simccY[k * widthY + i]
                if v > maxY {
                    maxY = v
                    argY = i
                }
            }
            let val = 0.5 * (maxX + maxY)
            var locX = Double(argX)
            var locY = Double(argY)
            if val <= 0 {
                locX = -1
                locY = -1
            }
            // locs / split_ratio / input_size * scale + center - scale/2
            let x = locX / simccSplitRatio / Double(inputWidth) * crop.scale.w
                + crop.center.x - crop.scale.w / 2
            let y = locY / simccSplitRatio / Double(inputHeight) * crop.scale.h
                + crop.center.y - crop.scale.h / 2
            out.append(Keypoint(x: x, y: y, conf: val))
        }
        return out
    }
}

// MARK: - Person selection (port of rtmpose_backend.select_person and
// detector.select_tracked_box)

public enum PersonSelection {
    /// Pick the athlete among detected people, or nil if nobody is credible.
    /// people: per-person keypoints. Prefers confidence, then proximity to the
    /// last known position (score − 0.5 · normalized distance).
    public static func selectPerson(people: [[Keypoint]],
                                    lastCenter: (x: Double, y: Double)?,
                                    minConf: Double,
                                    imageDiagonal: Double) -> Int? {
        let meanScores = people.map { kps in
            kps.map(\.conf).reduce(0, +) / Double(max(1, kps.count))
        }
        let ok = meanScores.indices.filter { meanScores[$0] >= minConf }
        guard !ok.isEmpty else { return nil }
        guard let last = lastCenter, imageDiagonal > 0 else {
            return ok.max { meanScores[$0] < meanScores[$1] }
        }
        func center(_ kps: [Keypoint]) -> (Double, Double) {
            let n = Double(max(1, kps.count))
            return (kps.map(\.x).reduce(0, +) / n, kps.map(\.y).reduce(0, +) / n)
        }
        return ok.max { a, b in
            func objective(_ i: Int) -> Double {
                let c = center(people[i])
                let dist = ((c.0 - last.x) * (c.0 - last.x)
                          + (c.1 - last.y) * (c.1 - last.y)).squareRoot()
                    / imageDiagonal
                return meanScores[i] - 0.5 * dist
            }
            return objective(a) < objective(b)
        }
    }

    /// Track a person box across frames: tallest first, then nearest to the
    /// previous pick (prevents jumping between people).
    public static func selectTrackedBox(_ boxes: [DetectionBox],
                                        lastCenter: (x: Double, y: Double)?) -> DetectionBox? {
        guard !boxes.isEmpty else { return nil }
        guard let last = lastCenter else {
            return boxes.max { $0.height < $1.height }
        }
        return boxes.min { a, b in
            func dist(_ box: DetectionBox) -> Double {
                let c = box.center
                return ((c.x - last.x) * (c.x - last.x)
                      + (c.y - last.y) * (c.y - last.y)).squareRoot()
            }
            return dist(a) < dist(b)
        }
    }
}
