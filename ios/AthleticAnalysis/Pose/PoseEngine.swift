// On-device pose estimation: AVAssetReader frames → YOLOX person detect →
// RTMPose Halpe-26, using ONNX Runtime (Core ML execution provider) with the
// same .onnx files rtmlib uses on desktop.
//
// Geometry/decoding math lives in AthleticAnalysisCore.PoseProcessing (parity-
// tested against rtmlib); this file owns pixels (vImage) and inference (ORT).
//
// Model files to bundle (from ~/.cache/rtmlib on the desktop, "balanced" tier):
//   det:  yolox_m_8xb8-300e_humanart  (input 640×640)
//   pose: rtmpose-m_simcc-body7_…-halpe26_256x192  (input 192×256, W×H)

import Accelerate
import AthleticAnalysisCore
import AVFoundation
import CoreVideo
import Foundation

// The ONNX Runtime SPM package (https://github.com/microsoft/onnxruntime-swift-package-manager)
// provides the `onnxruntime_objc` module. Guarded so the file parses before
// the dependency is added in Xcode.
#if canImport(onnxruntime_objc)
import onnxruntime_objc
#endif

final class PoseEngine {
    static let detInput = (w: 640, h: 640)
    static let poseInput = (w: 192, h: 256)
    static let detFrequency = 5  // re-detect every N frames, track in between
    static let minPersonConf = 0.35

    private var lastCenter: (x: Double, y: Double)?
    private var lastBoxCenter: (x: Double, y: Double)?

    #if canImport(onnxruntime_objc)
    private var env: ORTEnv?
    private var detSession: ORTSession?
    private var poseSession: ORTSession?

    private func ensureSessions() throws {
        guard detSession == nil else { return }
        let env = try ORTEnv(loggingLevel: .warning)
        self.env = env
        let options = try ORTSessionOptions()
        // Core ML EP with CPU fallback; harmless if unavailable (simulator).
        try? options.appendCoreMLExecutionProvider(with: ORTCoreMLExecutionProviderOptions())
        func session(_ name: String) throws -> ORTSession {
            guard let path = Bundle.main.path(forResource: name, ofType: "onnx") else {
                throw PoseEngineError.modelMissing(name)
            }
            return try ORTSession(env: env, modelPath: path, sessionOptions: options)
        }
        detSession = try session("det")
        poseSession = try session("pose")
    }
    #endif

    enum PoseEngineError: Error {
        case modelMissing(String)
        case notImplementedOnThisPlatform
    }

    /// Decode every frame of the asset and emit one Pose per frame, in order.
    func run(asset: AVAsset,
             onFrame: @escaping (Pose, Int) -> Void) async throws {
        #if canImport(onnxruntime_objc)
        try ensureSessions()
        #endif
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            return
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()

        var index = 0
        var trackedBox: DetectionBox?
        while let sample = output.copyNextSampleBuffer() {
            guard let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
            if index % Self.detFrequency == 0 || trackedBox == nil {
                let boxes = try detect(pixels)
                trackedBox = PersonSelection.selectTrackedBox(boxes,
                                                              lastCenter: lastBoxCenter)
                if let c = trackedBox?.center { lastBoxCenter = c }
            }
            let pose = try estimatePose(pixels, box: trackedBox)
            onFrame(pose, index)
            index += 1
        }
    }

    // MARK: - Detection

    private func detect(_ pixels: CVPixelBuffer) throws -> [DetectionBox] {
        let w = CVPixelBufferGetWidth(pixels)
        let h = CVPixelBufferGetHeight(pixels)
        let box = YOLOXProcessing.letterbox(imageWidth: w, imageHeight: h,
                                            inputWidth: Self.detInput.w,
                                            inputHeight: Self.detInput.h)
        let tensor = letterboxedTensor(pixels, letterbox: box,
                                       inputW: Self.detInput.w,
                                       inputH: Self.detInput.h)
        let (raw, cols) = try runDetModel(tensor)
        return YOLOXProcessing.decode(values: raw, cols: cols,
                                      inputWidth: Self.detInput.w,
                                      inputHeight: Self.detInput.h,
                                      ratio: box.ratio)
    }

    // MARK: - Pose

    private func estimatePose(_ pixels: CVPixelBuffer,
                              box: DetectionBox?) throws -> Pose {
        let w = CVPixelBufferGetWidth(pixels)
        let h = CVPixelBufferGetHeight(pixels)
        let bbox = box.map { (x1: $0.x1, y1: $0.y1, x2: $0.x2, y2: $0.y2) }
            ?? (x1: 0.0, y1: 0.0, x2: Double(w), y2: Double(h))
        let crop = RTMPoseProcessing.cropSpec(bbox: bbox,
                                              inputWidth: Self.poseInput.w,
                                              inputHeight: Self.poseInput.h)
        let tensor = warpedTensor(pixels, crop: crop,
                                  inputW: Self.poseInput.w,
                                  inputH: Self.poseInput.h)
        let (simccX, widthX, simccY, widthY) = try runPoseModel(tensor)
        let kpts = RTMPoseProcessing.decodeSimCC(simccX: simccX, widthX: widthX,
                                                 simccY: simccY, widthY: widthY,
                                                 keypointCount: KP.count,
                                                 crop: crop,
                                                 inputWidth: Self.poseInput.w,
                                                 inputHeight: Self.poseInput.h)
        return Pose(points: kpts)
    }

    // MARK: - Pixels → tensors (vImage)

    /// Letterbox-resize BGRA pixels onto a gray canvas and emit an NCHW
    /// float32 tensor of the BGR channels (rtmlib layout, no normalization).
    private func letterboxedTensor(_ pixels: CVPixelBuffer,
                                   letterbox: YOLOXProcessing.Letterbox,
                                   inputW: Int, inputH: Int) -> [Float] {
        var canvas = [UInt8](repeating: YOLOXProcessing.padValue,
                             count: inputW * inputH * 4)
        scaleBGRA(pixels, into: &canvas, canvasW: inputW, canvasH: inputH,
                  targetW: letterbox.resizedWidth, targetH: letterbox.resizedHeight)
        return bgraToNCHW(canvas, w: inputW, h: inputH,
                          mean: nil, std: nil)
    }

    /// Affine-warp the person crop to the model input and normalize.
    private func warpedTensor(_ pixels: CVPixelBuffer,
                              crop: RTMPoseProcessing.CropSpec,
                              inputW: Int, inputH: Int) -> [Float] {
        var canvas = [UInt8](repeating: 0, count: inputW * inputH * 4)
        warpBGRA(pixels, into: &canvas, canvasW: inputW, canvasH: inputH,
                 warp: crop.warp)
        return bgraToNCHW(canvas, w: inputW, h: inputH,
                          mean: RTMPoseProcessing.mean,
                          std: RTMPoseProcessing.std)
    }

    private func scaleBGRA(_ pixels: CVPixelBuffer, into canvas: inout [UInt8],
                           canvasW: Int, canvasH: Int,
                           targetW: Int, targetH: Int) {
        CVPixelBufferLockBaseAddress(pixels, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixels, .readOnly) }
        var src = vImage_Buffer(
            data: CVPixelBufferGetBaseAddress(pixels),
            height: vImagePixelCount(CVPixelBufferGetHeight(pixels)),
            width: vImagePixelCount(CVPixelBufferGetWidth(pixels)),
            rowBytes: CVPixelBufferGetBytesPerRow(pixels))
        var scaled = [UInt8](repeating: 0, count: targetW * targetH * 4)
        scaled.withUnsafeMutableBytes { buf in
            var dst = vImage_Buffer(data: buf.baseAddress,
                                    height: vImagePixelCount(targetH),
                                    width: vImagePixelCount(targetW),
                                    rowBytes: targetW * 4)
            vImageScale_ARGB8888(&src, &dst, nil, vImage_Flags(kvImageNoFlags))
        }
        for row in 0..<targetH {
            let srcStart = row * targetW * 4
            let dstStart = row * canvasW * 4
            canvas.replaceSubrange(dstStart..<(dstStart + targetW * 4),
                                   with: scaled[srcStart..<(srcStart + targetW * 4)])
        }
    }

    private func warpBGRA(_ pixels: CVPixelBuffer, into canvas: inout [UInt8],
                          canvasW: Int, canvasH: Int, warp: AffineMatrix) {
        CVPixelBufferLockBaseAddress(pixels, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixels, .readOnly) }
        var src = vImage_Buffer(
            data: CVPixelBufferGetBaseAddress(pixels),
            height: vImagePixelCount(CVPixelBufferGetHeight(pixels)),
            width: vImagePixelCount(CVPixelBufferGetWidth(pixels)),
            rowBytes: CVPixelBufferGetBytesPerRow(pixels))
        // vImage warps dst→src with a 3x3; invert our image→input matrix.
        // cv2.warpAffine samples src at M⁻¹·dst — same convention.
        let m = warp.m
        let det = m.0 * m.4 - m.1 * m.3
        let inv = (m.4 / det, -m.1 / det,
                   (m.1 * m.5 - m.4 * m.2) / det,
                   -m.3 / det, m.0 / det,
                   (m.3 * m.2 - m.0 * m.5) / det)
        var transform = vImage_AffineTransform(a: Float(inv.0), b: Float(inv.3),
                                               c: Float(inv.1), d: Float(inv.4),
                                               tx: Float(inv.2), ty: Float(inv.5))
        var bg: [UInt8] = [0, 0, 0, 255]
        canvas.withUnsafeMutableBytes { buf in
            var dst = vImage_Buffer(data: buf.baseAddress,
                                    height: vImagePixelCount(canvasH),
                                    width: vImagePixelCount(canvasW),
                                    rowBytes: canvasW * 4)
            _ = vImageAffineWarp_ARGB8888(&src, &dst, nil, &transform, &bg,
                                          vImage_Flags(kvImageBackgroundColorFill))
        }
    }

    /// BGRA bytes → (1, 3, H, W) float32 in B, G, R channel order
    /// (matches rtmlib: BGR image transposed to CHW, no channel swap).
    private func bgraToNCHW(_ bgra: [UInt8], w: Int, h: Int,
                            mean: (Double, Double, Double)?,
                            std: (Double, Double, Double)?) -> [Float] {
        var out = [Float](repeating: 0, count: 3 * w * h)
        let means = mean ?? (0, 0, 0)
        let stds = std ?? (1, 1, 1)
        let m = [Float(means.0), Float(means.1), Float(means.2)]
        let s = [Float(stds.0), Float(stds.1), Float(stds.2)]
        for y in 0..<h {
            for x in 0..<w {
                let p = (y * w + x) * 4  // B G R A
                for c in 0..<3 {
                    out[c * w * h + y * w + x] =
                        (Float(bgra[p + c]) - m[c]) / s[c]
                }
            }
        }
        return out
    }

    // MARK: - ORT inference

    private func runDetModel(_ tensor: [Float]) throws -> ([Double], Int) {
        #if canImport(onnxruntime_objc)
        let outputs = try runModel(detSession!, tensor: tensor,
                                   shape: [1, 3, NSNumber(value: Self.detInput.h),
                                           NSNumber(value: Self.detInput.w)])
        let raw = outputs[0]
        let cols = raw.shape.last!
        return (raw.values, cols)
        #else
        throw PoseEngineError.notImplementedOnThisPlatform
        #endif
    }

    private func runPoseModel(_ tensor: [Float]) throws
        -> (simccX: [Double], widthX: Int, simccY: [Double], widthY: Int) {
        #if canImport(onnxruntime_objc)
        let outputs = try runModel(poseSession!, tensor: tensor,
                                   shape: [1, 3, NSNumber(value: Self.poseInput.h),
                                           NSNumber(value: Self.poseInput.w)])
        let x = outputs[0]
        let y = outputs[1]
        return (x.values, x.shape.last!, y.values, y.shape.last!)
        #else
        throw PoseEngineError.notImplementedOnThisPlatform
        #endif
    }

    #if canImport(onnxruntime_objc)
    private struct ORTOutput {
        var values: [Double]
        var shape: [Int]
    }

    private func runModel(_ session: ORTSession, tensor: [Float],
                          shape: [NSNumber]) throws -> [ORTOutput] {
        let data = NSMutableData(bytes: tensor,
                                 length: tensor.count * MemoryLayout<Float>.stride)
        let input = try ORTValue(tensorData: data, elementType: .float,
                                 shape: shape)
        let inputName = try session.inputNames()[0]
        let outputNames = try session.outputNames()
        let results = try session.run(withInputs: [inputName: input],
                                      outputNames: Set(outputNames),
                                      runOptions: nil)
        return try outputNames.map { name in
            let value = results[name]!
            let info = try value.tensorTypeAndShapeInfo()
            let outShape = try info.shape.map { $0.intValue }
            let outData = try value.tensorData() as Data
            let floats = outData.withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
            return ORTOutput(values: floats.map { Double($0) }, shape: outShape)
        }
    }
    #endif
}
