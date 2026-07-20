// Parity tests for PoseProcessing.swift against rtmlib's real functions
// (fixtures from tools/export_pose_fixtures.py).

import Foundation
import XCTest
@testable import AthleticAnalysisCore

final class PoseProcessingTests: XCTestCase {
    static var fixture: [String: Any] = [:]

    override class func setUp() {
        super.setUp()
        guard let url = Bundle.module.url(forResource: "Fixtures/pose_processing",
                                          withExtension: "json")
            ?? Bundle.module.url(forResource: "pose_processing",
                                 withExtension: "json",
                                 subdirectory: "Fixtures"),
              let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return
        }
        fixture = obj
    }

    func d(_ any: Any?) -> Double {
        if any == nil || any is NSNull { return .nan }
        return (any as? NSNumber)?.doubleValue ?? .nan
    }

    func dArr(_ any: Any?) -> [Double] {
        (any as? [Any])?.map { d($0) } ?? []
    }

    func testCropSpecs() throws {
        let cases = Self.fixture["crop_specs"] as! [[String: Any]]
        XCTAssertFalse(cases.isEmpty)
        for c in cases {
            let bbox = dArr(c["bbox"])
            let spec = RTMPoseProcessing.cropSpec(
                bbox: (bbox[0], bbox[1], bbox[2], bbox[3]),
                inputWidth: Int(d(c["input_w"])), inputHeight: Int(d(c["input_h"])))
            let expCenter = dArr(c["center"])
            let expScale = dArr(c["scale"])
            XCTAssertEqual(spec.center.x, expCenter[0], accuracy: 1e-6)
            XCTAssertEqual(spec.center.y, expCenter[1], accuracy: 1e-6)
            XCTAssertEqual(spec.scale.w, expScale[0], accuracy: 1e-6)
            XCTAssertEqual(spec.scale.h, expScale[1], accuracy: 1e-6)
            let warpRows = (c["warp"] as! [Any]).map { dArr($0) }
            let m = spec.warp.m
            // cv2 works in float32; compare accordingly.
            XCTAssertEqual(m.0, warpRows[0][0], accuracy: 1e-3)
            XCTAssertEqual(m.1, warpRows[0][1], accuracy: 1e-3)
            XCTAssertEqual(m.2, warpRows[0][2], accuracy: 1e-2)
            XCTAssertEqual(m.3, warpRows[1][0], accuracy: 1e-3)
            XCTAssertEqual(m.4, warpRows[1][1], accuracy: 1e-3)
            XCTAssertEqual(m.5, warpRows[1][2], accuracy: 1e-2)
        }
    }

    func testSimCCDecode() throws {
        let cases = Self.fixture["simcc"] as! [[String: Any]]
        XCTAssertFalse(cases.isEmpty)
        for c in cases {
            let w = Int(d(c["input_w"]))
            let h = Int(d(c["input_h"]))
            let bbox = dArr(c["bbox"])
            let spec = RTMPoseProcessing.cropSpec(
                bbox: (bbox[0], bbox[1], bbox[2], bbox[3]),
                inputWidth: w, inputHeight: h)
            let simccX = (c["simcc_x"] as! [Any]).flatMap { dArr($0) }
            let simccY = (c["simcc_y"] as! [Any]).flatMap { dArr($0) }
            let kpts = RTMPoseProcessing.decodeSimCC(
                simccX: simccX, widthX: w * 2,
                simccY: simccY, widthY: h * 2,
                keypointCount: 26, crop: spec, inputWidth: w, inputHeight: h)
            let expKpts = (c["expected_kpts"] as! [Any]).map { dArr($0) }
            let expScores = dArr(c["expected_scores"])
            XCTAssertEqual(kpts.count, 26)
            for k in 0..<26 {
                XCTAssertEqual(kpts[k].x, expKpts[k][0], accuracy: 1e-3,
                               "kp\(k).x")
                XCTAssertEqual(kpts[k].y, expKpts[k][1], accuracy: 1e-3,
                               "kp\(k).y")
                XCTAssertEqual(kpts[k].conf, expScores[k], accuracy: 1e-5,
                               "kp\(k).score")
            }
        }
    }

    func testYOLOXDecode() throws {
        let cases = Self.fixture["yolox"] as! [[String: Any]]
        XCTAssertFalse(cases.isEmpty)
        for c in cases {
            let cols = Int(d(c["cols"]))
            let raw = (c["raw"] as! [Any]).flatMap { dArr($0) }
            let boxes = YOLOXProcessing.decode(
                values: raw, cols: cols,
                inputWidth: Int(d(c["input_w"])), inputHeight: Int(d(c["input_h"])),
                ratio: d(c["ratio"]))
            let expected = (c["expected_boxes"] as! [Any]).map { dArr($0) }
            XCTAssertEqual(boxes.count, expected.count, "box count")
            for (b, e) in zip(boxes, expected) {
                XCTAssertEqual(b.x1, e[0], accuracy: 1e-2)
                XCTAssertEqual(b.y1, e[1], accuracy: 1e-2)
                XCTAssertEqual(b.x2, e[2], accuracy: 1e-2)
                XCTAssertEqual(b.y2, e[3], accuracy: 1e-2)
            }
        }
    }

    func testPersonSelection() throws {
        let cases = Self.fixture["selection"] as! [[String: Any]]
        XCTAssertFalse(cases.isEmpty)
        for c in cases {
            let people = (c["keypoints"] as! [Any]).enumerated().map { i, kps -> [Keypoint] in
                let coords = (kps as! [Any]).map { dArr($0) }
                let scores = dArr((c["scores"] as! [Any])[i])
                return zip(coords, scores).map {
                    Keypoint(x: $0.0[0], y: $0.0[1], conf: $0.1)
                }
            }
            var last: (x: Double, y: Double)? = nil
            if let lc = c["last_center"] as? [Any] {
                last = (d(lc[0]), d(lc[1]))
            }
            let got = PersonSelection.selectPerson(
                people: people, lastCenter: last,
                minConf: d(c["min_conf"]), imageDiagonal: d(c["img_diag"]))
            let expected = Int(d(c["expected_index"]))
            XCTAssertEqual(got ?? -1, expected)
        }
    }
}
