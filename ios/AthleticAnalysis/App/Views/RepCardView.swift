// Rep Card: the first thing shown after analysis. Clip quality, the rep's
// phase shape, hero numbers, the sprint-factor pentagon, a coaching summary,
// and the top issues to fix — all in one scroll, each finding tappable to
// jump to its frame (port of the desktop ui/rep_card.py).

import AthleticAnalysisCore
import SwiftUI

struct RepCardView: View {
    let analysis: SprintAnalysis
    let fps: Double
    var onSeek: (Int) -> Void = { _ in }

    private func fmt(_ v: Double, _ decimals: Int = 2) -> String {
        v.isFinite ? String(format: "%.\(decimals)f", v) : "–"
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                qualityCard
                phaseOverview
                heroGrid
                if let radar = analysis.radar {
                    VStack(alignment: .leading, spacing: 10) {
                        SectionHeader(title: "Sprint factor profile")
                        RadarView(radar: radar)
                    }
                    .cardStyle()
                }
                summaryBanner
                VStack(alignment: .leading, spacing: 10) {
                    SectionHeader(title: "Top issues")
                    topIssues
                }
            }
            .padding()
        }
        .background(Theme.background)
    }

    // MARK: - Sections

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Rep Card").font(.system(.largeTitle, design: .rounded).weight(.bold))
            if let level = analysis.radar?.level {
                Text("Graded for \(level.rawValue.capitalized) level · "
                     + "\(analysis.metrics.steps.count) steps analyzed")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var qualityCard: some View {
        let q = analysis.quality
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Analysis quality").font(Theme.sectionTitle)
                Spacer()
                Chip(text: "\(q.level.rawValue) confidence", color: Theme.confidence(q.level),
                     filled: true)
            }
            ForEach(q.notes, id: \.self) { note in
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Circle().fill(Color.secondary).frame(width: 3, height: 3).padding(.top, 5)
                    Text(note).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        .cardStyle()
    }

    private var phaseOverview: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionHeader(title: "Shape of this rep")
            PhaseRibbon(spans: Coaching.segmentPhases(analysis.velocities.runSpeed, fps: fps),
                        totalFrames: analysis.keypoints.count, height: 14)
            PhaseLegend()
        }
        .cardStyle()
    }

    private var heroGrid: some View {
        let m = analysis.metrics
        let unit = m.lengthUnit
        return LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
            heroTile(icon: "bolt.fill", value: "\(fmt(m.maxSpeed)) \(analysis.velocities.unit)",
                     caption: "top speed", accent: Theme.accent)
            heroTile(icon: "waveform.path.ecg", value: fmt(m.cadenceSpm, 0),
                     caption: "steps / min", accent: Theme.phase(.acceleration))
            heroTile(icon: "timer", value: "\(fmt(m.meanContactS * 1000, 0)) ms",
                     caption: "avg ground contact", accent: Theme.phase(.drive))
            heroTile(icon: "arrow.up.and.down", value: "\(fmt(m.meanFlightS * 1000, 0)) ms",
                     caption: "avg flight time", accent: Theme.phase(.maxVelocity))
            heroTile(icon: "arrow.left.and.right", value: "\(fmt(m.meanStepLength)) \(unit)",
                     caption: "avg step length", accent: Theme.phase(.deceleration))
            heroTile(icon: "figure.stand", value: "\(fmt(m.meanTrunkLeanDeg, 0))°",
                     caption: "avg trunk lean", accent: Theme.accent)
        }
    }

    private func heroTile(icon: String, value: String, caption: String,
                          accent: Color) -> some View {
        HStack(alignment: .top, spacing: 10) {
            RoundedRectangle(cornerRadius: 2).fill(accent).frame(width: 4)
            VStack(alignment: .leading, spacing: 3) {
                Image(systemName: icon).font(.caption).foregroundStyle(accent)
                Text(value).font(Theme.hero(21)).lineLimit(1).minimumScaleFactor(0.7)
                Text(caption).font(.caption).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(Theme.surfaceRaised, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    private var summaryBanner: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "clipboard.fill").foregroundStyle(Theme.accent)
            Text(Coaching.summarize(analysis.findings)).font(.subheadline)
        }
        .cardStyle()
    }

    private var topIssues: some View {
        let faults = analysis.findings.filter { $0.severity != .good }
        return VStack(alignment: .leading, spacing: 8) {
            if faults.isEmpty {
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.seal.fill").foregroundStyle(Theme.good)
                    Text("No form faults detected in this rep").font(.subheadline)
                }
                .cardStyle()
            } else {
                ForEach(Array(faults.prefix(3).enumerated()), id: \.offset) { _, f in
                    Button {
                        onSeek(f.frame)
                    } label: {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Chip(text: f.severity == .major ? "Major" : "Minor",
                                     color: Theme.severity(f.severity), filled: true)
                                Chip(text: f.phase.capitalized, color: Theme.phase(named: f.phase))
                                Spacer()
                                if let conf = f.confidence, !conf.limiter.isEmpty {
                                    Chip(text: conf.level.rawValue, color: Theme.confidence(conf.level))
                                }
                            }
                            Text(f.metric).font(.subheadline.weight(.semibold))
                            Text("\(f.valueText) vs optimal \(f.targetText)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .buttonStyle(.plain)
                    .cardStyle()
                }
            }
        }
    }
}
