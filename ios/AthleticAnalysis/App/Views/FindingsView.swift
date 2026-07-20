// Form findings: every check the coach graded, with expandable root-cause
// diagnostics (port of the desktop ui/form_panel.py + diagnosis pane).

import AthleticAnalysisCore
import SwiftUI

struct FindingsView: View {
    let findings: [FormFinding]
    var onSeek: (Int) -> Void = { _ in }

    private enum Filter: String, CaseIterable { case all = "All checks", issues = "Issues only" }

    @State private var filter: Filter = .all
    // Keyed by "phase|metric" (stable across filtering — each (phase, metric)
    // pair is unique within one analysis run) rather than array offset.
    @State private var expanded: Set<String> = []

    private var visible: [FormFinding] {
        filter == .all ? findings : findings.filter { $0.severity != .good }
    }

    var body: some View {
        VStack(spacing: 0) {
            Picker("Filter", selection: $filter) {
                ForEach(Filter.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .padding()
            .background(Theme.background)

            if visible.isEmpty {
                ContentUnavailableFallback()
            } else {
                List {
                    ForEach(Array(visible.enumerated()), id: \.offset) { _, finding in
                        Section {
                            row(finding)
                            if isExpanded(finding) {
                                if let diag = Diagnostics.diagnose(finding) {
                                    DiagnosisView(finding: finding, diagnosis: diag)
                                } else if finding.severity != .good {
                                    Text("No root-cause entry for this deviation — it usually "
                                         + "indicates a tracking or capture-FPS issue rather "
                                         + "than a form fault.")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                        .listRowBackground(Theme.surface)
                    }
                }
                .listStyle(.insetGrouped)
                .scrollContentBackground(.hidden)
                .background(Theme.background)
            }
        }
        .background(Theme.background)
    }

    private func key(_ f: FormFinding) -> String { "\(f.phase)|\(f.metric)" }
    private func isExpanded(_ f: FormFinding) -> Bool { expanded.contains(key(f)) }

    private func row(_ finding: FormFinding) -> some View {
        Button {
            onSeek(finding.frame)
            guard finding.severity != .good else { return }
            let k = key(finding)
            if expanded.contains(k) { expanded.remove(k) } else { expanded.insert(k) }
        } label: {
            HStack(alignment: .top, spacing: 10) {
                Circle()
                    .fill(Theme.severity(finding.severity))
                    .frame(width: 9, height: 9)
                    .padding(.top, 4)
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Chip(text: finding.phase.capitalized,
                             color: Theme.phase(named: finding.phase))
                        if let conf = finding.confidence, !conf.limiter.isEmpty {
                            Chip(text: "\(conf.level.rawValue) · \(conf.limiter)",
                                 color: Theme.confidence(conf.level))
                        }
                    }
                    Text(finding.metric).font(.subheadline.weight(.medium))
                    Text("\(finding.valueText) vs optimal \(finding.targetText)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if finding.severity != .good {
                    Image(systemName: isExpanded(finding) ? "chevron.up" : "chevron.down")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .buttonStyle(.plain)
    }
}

private struct ContentUnavailableFallback: View {
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "checkmark.seal.fill")
                .font(.largeTitle)
                .foregroundStyle(Theme.good)
            Text("No issues in this rep").font(.headline)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.background)
    }
}

struct DiagnosisView: View {
    let finding: FormFinding
    let diagnosis: Diagnosis

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(Theme.severity(finding.severity))
                    .frame(width: 3)
                Text(diagnosis.title)
                    .font(.subheadline.bold())
                    .foregroundStyle(Theme.severity(finding.severity))
            }
            section("Why this happens", diagnosis.technicalCauses)
            section("Likely physical limiters", diagnosis.muscleFactors)
            section("Corrective drills", diagnosis.drills)
            if !diagnosis.phaseNote.isEmpty {
                Text(diagnosis.phaseNote)
                    .font(.caption.italic())
                    .foregroundStyle(.secondary)
            }
            if !diagnosis.source.isEmpty {
                Text("Source: \(diagnosis.source)")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(10)
        .background(Theme.surfaceRaised, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .padding(.vertical, 4)
    }

    private func section(_ title: String, _ items: [String]) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.caption.bold())
            ForEach(items, id: \.self) { item in
                HStack(alignment: .firstTextBaseline, spacing: 5) {
                    Text("•")
                    Text(item)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
    }
}
