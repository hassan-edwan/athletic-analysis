// The app's visual signature: a run's phase composition (drive →
// acceleration → max velocity → deceleration) drawn as one tinted ribbon.
// The exact same shape and colors appear as the video scrubber background,
// the Rep Card overview strip, and the shading behind the per-step charts —
// wherever a rep is shown, its phase structure reads the same way.

import AthleticAnalysisCore
import SwiftUI

struct PhaseRibbon: View {
    let spans: [(Int, Int, SprintPhase)]
    let totalFrames: Int
    var height: CGFloat = 10
    var cornerRadius: CGFloat = 4

    var body: some View {
        GeometryReader { geo in
            let total = max(1, totalFrames - 1)
            ZStack(alignment: .topLeading) {
                Rectangle().fill(Theme.surfaceRaised)
                ForEach(Array(spans.enumerated()), id: \.offset) { _, span in
                    let (start, end, phase) = span
                    let x0 = geo.size.width * CGFloat(start) / CGFloat(total)
                    let x1 = geo.size.width * CGFloat(end + 1) / CGFloat(total)
                    Rectangle()
                        .fill(Theme.phase(phase).opacity(0.85))
                        .frame(width: max(1, x1 - x0))
                        .offset(x: x0)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        }
        .frame(height: height)
    }
}

/// Small dot + label legend for the four sprint phases, used under any
/// PhaseRibbon so the colors are always explained on first read.
struct PhaseLegend: View {
    var phases: [SprintPhase] = [.drive, .acceleration, .maxVelocity, .deceleration]

    var body: some View {
        HStack(spacing: 12) {
            ForEach(phases, id: \.self) { phase in
                HStack(spacing: 4) {
                    Circle().fill(Theme.phase(phase)).frame(width: 6, height: 6)
                    Text(phase.rawValue.capitalized)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}
