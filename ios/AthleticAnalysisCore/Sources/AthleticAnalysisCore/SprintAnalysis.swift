// Full sprint pipeline (mirrors AnalysisSession.recompute()'s sprint path):
// raw keypoints → smooth → angles → velocities → events → metrics → findings
// → radar → clip quality. The app recomputes from raw keypoints on load and on
// level change, exactly like the desktop sidecar model.

import Foundation

public struct SprintAnalysis: Sendable {
    public var keypoints: PoseSequence  // smoothed
    public var angles: [String: [Double]]
    public var velocities: Velocities
    public var gaitEvents: [GaitEvent]
    public var metrics: SprintMetrics
    public var findings: [FormFinding]
    public var radar: SprintRadar?
    public var quality: ClipQuality

    public static func analyze(rawKeypoints: PoseSequence, fps: Double,
                               level: AthleteLevel = .trained,
                               calib: Calibration? = nil) -> SprintAnalysis {
        let smoothed = Filtering.smoothKeypoints(rawKeypoints, fps: fps)
        let angles = Angles.computeAngles(smoothed)
        let velocities = Velocity.computeVelocities(smoothed, fps: fps, calib: calib)
        let events = GaitEvents.detectGaitEvents(smoothed, fps: fps)
        let metrics = SprintMetricsComputer.compute(smoothed, angles: angles,
                                                    events: events, fps: fps,
                                                    calib: calib)
        let findings = Coaching.analyzeSprintForm(smoothed, sprint: metrics,
                                                  runSpeed: velocities.runSpeed,
                                                  fps: fps, level: level)
        let radar = Radar.computeSprintRadar(smoothed, sprint: metrics,
                                             runSpeed: velocities.runSpeed,
                                             fps: fps, level: level)
        let quality = Confidence.clipQuality(smoothed, fps: fps,
                                             calibrated: calib != nil)
        return SprintAnalysis(keypoints: smoothed, angles: angles,
                              velocities: velocities, gaitEvents: events,
                              metrics: metrics, findings: findings,
                              radar: radar, quality: quality)
    }
}
