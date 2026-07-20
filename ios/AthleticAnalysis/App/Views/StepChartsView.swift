// Per-step bar charts: ground contact time, step length, step speed — colored
// by leg, with the same PhaseRibbon strip used everywhere else marking which
// sprint phase each step fell in (port of the desktop ui/step_charts.py).

import AthleticAnalysisCore
import Charts
import SwiftUI

struct StepChartsView: View {
    let analysis: SprintAnalysis
    let fps: Double

    private struct StepPoint: Identifiable {
        let id: Int
        let side: GaitEvent.Side
        let phase: SprintPhase?
        let value: Double
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader(title: "Per-step trends")
            HStack(spacing: 14) {
                legendDot(Theme.legLeft, "Left")
                legendDot(Theme.legRight, "Right")
                Spacer()
                PhaseLegend()
            }
            .font(.caption)

            chartCard(title: "Ground contact (ms)", metric: \.contactTimeS, scale: 1000)
            chartCard(title: "Step length (\(analysis.metrics.lengthUnit))", metric: \.stepLength)
            chartCard(title: "Step speed (\(analysis.metrics.lengthUnit)/s)", metric: \.stepSpeed)
        }
    }

    // MARK: - Data

    /// Strike-frame → phase, from the same bucketing the coach and radar use,
    /// so a step's phase color here never disagrees with Findings or Radar.
    private var stepPhase: [Int: SprintPhase] {
        let buckets = Coaching.bucketSprintSteps(analysis.keypoints, sprint: analysis.metrics,
                                                 runSpeed: analysis.velocities.runSpeed, fps: fps)
        var map: [Int: SprintPhase] = [:]
        for (phase, bucket) in buckets {
            for frame in bucket.strikeFrames { map[frame] = phase }
        }
        return map
    }

    private func points(_ metric: KeyPath<StepRecord, Double>, scale: Double = 1) -> [StepPoint] {
        let phases = stepPhase
        return analysis.metrics.steps.enumerated().compactMap { i, step in
            let v = step[keyPath: metric] * scale
            guard v.isFinite else { return nil }
            return StepPoint(id: i, side: step.side, phase: phases[step.strikeFrame], value: v)
        }
    }

    /// Contiguous same-phase runs over step index (not frame index) — feeds
    /// the little PhaseRibbon strip above each chart.
    private func phaseBands(_ data: [StepPoint]) -> [(Int, Int, SprintPhase)] {
        var spans: [(Int, Int, SprintPhase)] = []
        for p in data {
            guard let phase = p.phase else { continue }
            if let last = spans.last, last.2 == phase, last.1 == p.id - 1 {
                spans[spans.count - 1].1 = p.id
            } else {
                spans.append((p.id, p.id, phase))
            }
        }
        return spans
    }

    // MARK: - View pieces

    private func legendDot(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(label).foregroundStyle(.secondary)
        }
    }

    private func chartCard(title: String, metric: KeyPath<StepRecord, Double>,
                           scale: Double = 1) -> some View {
        let data = points(metric, scale: scale)
        return VStack(alignment: .leading, spacing: 8) {
            Text(title).font(Theme.sectionTitle)
            if data.isEmpty {
                Text("Not enough steps to chart.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(height: 80)
            } else {
                PhaseRibbon(spans: phaseBands(data), totalFrames: data.count, height: 6)
                Chart(data) { p in
                    BarMark(x: .value("Step", p.id), y: .value(title, p.value))
                        .foregroundStyle(Theme.leg(p.side))
                        .cornerRadius(3)
                }
                .chartXAxis(.hidden)
                .frame(height: 120)
            }
        }
        .cardStyle()
    }
}
