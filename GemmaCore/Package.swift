// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "GemmaCore",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "GemmaCore", targets: ["GemmaCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "0.1.12"),
    ],
    targets: [
        .target(
            name: "GemmaCore",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
            ]
        ),
    ]
)
