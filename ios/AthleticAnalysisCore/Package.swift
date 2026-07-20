// swift-tools-version: 5.9
// AthleticAnalysisCore — pure-Swift port of the Python sprint-analysis core.
// No UIKit/CoreML/Accelerate: builds and tests on any platform with the Swift
// toolchain (including Windows), so the analysis math is developed and proven
// against Python golden fixtures before ever touching Xcode.
import PackageDescription

let package = Package(
    name: "AthleticAnalysisCore",
    products: [
        .library(name: "AthleticAnalysisCore", targets: ["AthleticAnalysisCore"]),
    ],
    targets: [
        .target(name: "AthleticAnalysisCore"),
        .testTarget(
            name: "AthleticAnalysisCoreTests",
            dependencies: ["AthleticAnalysisCore"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
