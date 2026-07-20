// Shared visual language for every screen: palette, typography, and small
// reusable components (cards, chips, the phase ribbon). Data-semantic colors
// (phase/severity/confidence) reuse the exact RGB values the desktop app
// already settled on (ui/plot_panel.py PHASE_COLORS, ui/rep_card.py
// _CONF_COLOR, ui/radar_widget.py score bands) so the two apps read as one
// product instead of two independent skins. Only the brand accent (used for
// buttons, the selected tab, progress) is iOS-specific — chosen to sit apart
// from every data color so "chrome" and "data" never get confused.

import AthleticAnalysisCore
import SwiftUI

enum Theme {
    // MARK: - Surfaces

    static let background = Color(red: 0.043, green: 0.047, blue: 0.058)   // #0B0C0F
    static let surface = Color(red: 0.098, green: 0.102, blue: 0.125)      // #191A20
    static let surfaceRaised = Color(red: 0.133, green: 0.137, blue: 0.161) // #22232A
    static let hairline = Color.white.opacity(0.08)

    /// Brand accent — a cool "timing display" blue, distinct from every
    /// data-semantic color below so chrome never reads as data.
    static let accent = Color(red: 0.23, green: 0.61, blue: 1.0)  // #3B9CFF

    // MARK: - Data semantics (verbatim from the desktop palette)

    static let good = Color(red: 0.24, green: 0.64, blue: 0.36)  // #3DA35D
    static let warn = Color(red: 0.79, green: 0.59, blue: 0.10)  // #C9971A
    static let bad = Color(red: 0.82, green: 0.27, blue: 0.24)   // #D0453C

    static let legLeft = Color(red: 0.31, green: 0.78, blue: 0.31)
    static let legRight = Color(red: 1.0, green: 0.55, blue: 0.24)

    static func severity(_ s: Severity) -> Color {
        switch s {
        case .major: return bad
        case .minor: return warn
        case .good: return good
        }
    }

    static func confidence(_ level: MetricConfidence.Level) -> Color {
        switch level {
        case .high: return good
        case .medium: return warn
        case .low: return .secondary
        }
    }

    static func leg(_ side: GaitEvent.Side) -> Color {
        side == .left ? legLeft : legRight
    }

    /// Sprint-phase tint, matching desktop `PHASE_COLORS`.
    static func phase(_ phase: SprintPhase) -> Color {
        switch phase {
        case .drive: return Color(red: 1.0, green: 0.47, blue: 0.24)        // #FF783C
        case .acceleration: return Color(red: 1.0, green: 0.78, blue: 0.24) // #FFC83C
        case .maxVelocity: return Color(red: 0.31, green: 0.86, blue: 0.47) // #50DC78
        case .deceleration: return Color(red: 0.59, green: 0.59, blue: 0.86) // #9696DC
        }
    }

    static func phase(named name: String) -> Color {
        SprintPhase(rawValue: name).map(phase) ?? .secondary
    }

    // MARK: - Typography

    /// Big numeric readouts (hero tiles, live speed). Rounded design gives the
    /// stats a friendlier, more athletic feel than plain system numerals;
    /// tabular digits keep them from jittering in width as they update.
    static func hero(_ size: CGFloat = 26) -> Font {
        .system(size: size, weight: .bold, design: .rounded).monospacedDigit()
    }

    static let sectionTitle = Font.system(.headline, design: .rounded).weight(.semibold)
    static let label = Font.system(.subheadline, design: .rounded).weight(.medium)

    // MARK: - Layout

    static let cardRadius: CGFloat = 18
}

// MARK: - Reusable components

/// A small colored pill label — severity, confidence, or phase tags.
struct Chip: View {
    var text: String
    var color: Color
    var filled: Bool = false

    var body: some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .lineLimit(1)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(filled ? color : color.opacity(0.16), in: Capsule())
            .foregroundStyle(filled ? Color.black.opacity(0.85) : color)
    }
}

private struct CardBackground: ViewModifier {
    var padding: CGFloat

    func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(Theme.surface, in: RoundedRectangle(cornerRadius: Theme.cardRadius,
                                                             style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
                    .strokeBorder(Theme.hairline, lineWidth: 1)
            )
    }
}

extension View {
    /// The app's standard card surface: flat dark panel, hairline border, no
    /// drop shadow (shadows read as clutter stacked on an already-dark page).
    func cardStyle(padding: CGFloat = 14) -> some View {
        modifier(CardBackground(padding: padding))
    }
}

/// A section title with optional trailing accessory, used above every card
/// group so the hierarchy is scannable in a long results screen.
struct SectionHeader: View {
    var title: String
    var trailing: String? = nil

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title).font(Theme.sectionTitle)
            Spacer()
            if let trailing {
                Text(trailing).font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}
