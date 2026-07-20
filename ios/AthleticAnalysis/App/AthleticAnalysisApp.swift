// App entry point. NOTE: files under ios/AthleticAnalysis are the Xcode app
// target — they compile only on macOS/Xcode (SwiftUI + AVFoundation + ONNX
// Runtime). Written ahead on Windows; expect small compile fixes in the first
// Mac session.

import SwiftUI

@main
struct AthleticAnalysisApp: App {
    @State private var store = AnalysisStore()

    var body: some Scene {
        WindowGroup {
            HomeView()
                .environment(store)
                .preferredColorScheme(.dark)
        }
    }
}
