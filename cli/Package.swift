// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "GemmaChatCLI",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(path: "../GemmaCore"),
    ],
    targets: [
        .executableTarget(
            name: "GemmaChatCLI",
            dependencies: [
                .product(name: "GemmaCore", package: "GemmaCore"),
            ]
        ),
    ]
)
