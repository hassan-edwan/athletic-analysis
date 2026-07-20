// Home: import a sprint clip, run analysis, then browse Video / Summary /
// Steps / Form tabs.

import AthleticAnalysisCore
import PhotosUI
import SwiftUI

struct HomeView: View {
    @Environment(AnalysisStore.self) private var store
    @State private var pickedItem: PhotosPickerItem?
    @State private var currentFrame = 0

    var body: some View {
        @Bindable var store = store
        NavigationStack {
            Group {
                switch store.phase {
                case .idle:
                    importPrompt
                case .importing:
                    statusScreen(title: "Loading video…", systemImage: "square.and.arrow.down")
                case .analyzing(let progress):
                    analyzingScreen(progress: progress)
                case .failed(let message):
                    failedScreen(message: message)
                case .ready:
                    resultTabs
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Theme.background)
            .navigationTitle("Athletic Analysis")
            .toolbar {
                if case .ready = store.phase {
                    ToolbarItem(placement: .topBarLeading) {
                        Button {
                            currentFrame = 0
                            store.reset()
                        } label: {
                            Label("New Clip", systemImage: "arrow.counterclockwise")
                        }
                    }
                    ToolbarItem(placement: .topBarTrailing) {
                        Picker("Level", selection: $store.level) {
                            ForEach(AthleteLevel.allCases, id: \.self) { level in
                                Text(level.rawValue.capitalized).tag(level)
                            }
                        }
                        .tint(Theme.accent)
                    }
                }
            }
        }
        .tint(Theme.accent)
        .onChange(of: pickedItem) { _, item in
            guard let item else { return }
            Task {
                if let movie = try? await item.loadTransferable(type: VideoFile.self) {
                    await store.importVideo(url: movie.url)
                }
            }
        }
    }

    // MARK: - States

    private var importPrompt: some View {
        VStack(spacing: 20) {
            Spacer()
            ZStack {
                Circle()
                    .fill(Theme.accent.opacity(0.16))
                    .frame(width: 132, height: 132)
                    .blur(radius: 2)
                Image(systemName: "figure.run")
                    .font(.system(size: 52, weight: .semibold))
                    .foregroundStyle(Theme.accent)
            }
            VStack(spacing: 6) {
                Text("Analyze a sprint").font(.system(.title2, design: .rounded).weight(.bold))
                Text("Film from the side, whole body in frame.\nSlow-mo (120–240 fps) gives the best timing.")
                    .multilineTextAlignment(.center)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            PhotosPicker("Choose Video", selection: $pickedItem, matching: .videos)
                .font(.headline)
                .buttonStyle(.borderedProminent)
                .tint(Theme.accent)
            Spacer()
            Spacer()
        }
        .padding()
    }

    private func statusScreen(title: String, systemImage: String) -> some View {
        VStack(spacing: 14) {
            Image(systemName: systemImage)
                .font(.system(size: 34))
                .foregroundStyle(Theme.accent)
            Text(title).foregroundStyle(.secondary)
        }
    }

    private func analyzingScreen(progress: Double) -> some View {
        VStack(spacing: 14) {
            ProgressView(value: progress)
                .tint(Theme.accent)
                .padding(.horizontal, 40)
            Text("Analyzing pose… \(Int(progress * 100))%")
                .font(Theme.hero(20))
            Text("Runs once per clip — results are cached for next time.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
    }

    private func failedScreen(message: String) -> some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 34))
                .foregroundStyle(Theme.bad)
            Text(message)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 32)
            PhotosPicker("Try Another Video", selection: $pickedItem, matching: .videos)
                .buttonStyle(.borderedProminent)
                .tint(Theme.accent)
            Spacer()
            Spacer()
        }
    }

    @ViewBuilder
    private var resultTabs: some View {
        if let analysis = store.analysis, let url = store.videoURL {
            TabView {
                PlayerView(videoURL: url, analysis: analysis,
                           fps: store.fps, currentFrame: $currentFrame)
                    .tabItem { Label("Video", systemImage: "play.rectangle.fill") }
                RepCardView(analysis: analysis, fps: store.fps) { currentFrame = $0 }
                    .tabItem { Label("Summary", systemImage: "chart.bar.fill") }
                StepsView(videoURL: url, analysis: analysis, fps: store.fps) { currentFrame = $0 }
                    .tabItem { Label("Steps", systemImage: "shoeprints.fill") }
                FindingsView(findings: analysis.findings) { currentFrame = $0 }
                    .tabItem { Label("Form", systemImage: "checklist") }
            }
        } else {
            VStack(spacing: 10) {
                Image(systemName: "questionmark.circle")
                    .font(.system(size: 34))
                    .foregroundStyle(.secondary)
                Text("No sprint steps detected in this clip.")
                    .foregroundStyle(.secondary)
                PhotosPicker("Try Another Video", selection: $pickedItem, matching: .videos)
                    .buttonStyle(.bordered)
                    .tint(Theme.accent)
            }
        }
    }
}

/// Transferable wrapper copying the picked video into a temporary file.
struct VideoFile: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { file in
            SentTransferredFile(file.url)
        } importing: { received in
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + ".mov")
            try FileManager.default.copyItem(at: received.file, to: dest)
            return VideoFile(url: dest)
        }
    }
}
