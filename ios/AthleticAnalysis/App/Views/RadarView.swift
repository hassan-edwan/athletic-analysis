// Pentagon (radar) chart of the five sprint-factor scores.
// SwiftUI Canvas port of the desktop ui/radar_widget.py drawing.

import AthleticAnalysisCore
import SwiftUI

struct RadarView: View {
    let radar: SprintRadar

    private let shortNames = ["Stiffness", "Front-side", "Posture",
                              "Foot placement", "Rhythm"]

    private func scoreColor(_ score: Double) -> Color {
        guard score.isFinite else { return .secondary }
        if score < 60 { return Theme.bad }
        if score < 85 { return Theme.warn }
        return Theme.good
    }

    private func vertex(center: CGPoint, radius: CGFloat, axis: Int,
                        r: Double) -> CGPoint {
        let angle = -Double.pi / 2 + 2 * .pi * Double(axis) / 5
        return CGPoint(x: center.x + radius * CGFloat(r / 100 * cos(angle)),
                       y: center.y + radius * CGFloat(r / 100 * sin(angle)))
    }

    var body: some View {
        Canvas { ctx, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2 + 4)
            let radius = max(10, min(size.width / 2 - 76, size.height / 2 - 24))

            // Grid rings + spokes (recessive).
            for ring in stride(from: 20.0, through: 100.0, by: 20.0) {
                var path = Path()
                path.addLines((0..<5).map {
                    vertex(center: center, radius: radius, axis: $0, r: ring)
                })
                path.closeSubpath()
                ctx.stroke(path, with: .color(.secondary.opacity(0.25)),
                           lineWidth: 0.5)
            }
            for axis in 0..<5 {
                var spoke = Path()
                spoke.move(to: center)
                spoke.addLine(to: vertex(center: center, radius: radius,
                                         axis: axis, r: 100))
                ctx.stroke(spoke, with: .color(.secondary.opacity(0.35)),
                           lineWidth: 0.5)
            }

            // Score polygon (NaN axes collapse to the center).
            let accent = Theme.accent
            let points = radar.axes.enumerated().map { i, axis in
                vertex(center: center, radius: radius, axis: i,
                       r: axis.score.isFinite ? axis.score : 0)
            }
            var poly = Path()
            poly.addLines(points)
            poly.closeSubpath()
            ctx.fill(poly, with: .color(accent.opacity(0.28)))
            ctx.stroke(poly, with: .color(accent), lineWidth: 2)
            for (i, p) in points.enumerated() where radar.axes[i].score.isFinite {
                let dot = Path(ellipseIn: CGRect(x: p.x - 3, y: p.y - 3,
                                                 width: 6, height: 6))
                ctx.fill(dot, with: .color(accent))
            }

            // Labels + numeric scores outside each vertex.
            for (i, axis) in radar.axes.enumerated() {
                let anchor = vertex(center: center, radius: radius, axis: i,
                                    r: 122)
                let scoreText = axis.score.isFinite
                    ? "\(Int(axis.score.rounded()))" : "n/a"
                let label = Text("\(shortNames[i]) ").font(.caption2)
                    .foregroundStyle(.secondary)
                    + Text(scoreText).font(.caption2.bold())
                    .foregroundStyle(scoreColor(axis.score))
                ctx.draw(label, at: anchor)
            }

            // Overall in the middle.
            if radar.overall.isFinite {
                ctx.draw(Text("overall \(Int(radar.overall.rounded()))")
                    .font(.caption2).foregroundStyle(.secondary),
                    at: center)
            }
        }
        .frame(height: 230)
        .accessibilityElement()
        .accessibilityLabel(accessibilitySummary)
    }

    private var accessibilitySummary: String {
        radar.axes.map { axis in
            let s = axis.score.isFinite
                ? "\(Int(axis.score.rounded())) out of 100" : "no data"
            return "\(axis.name): \(s)"
        }.joined(separator: ", ")
    }
}
