// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "GemmaBench",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(path: "../../../GemmaCore"),
    ],
    targets: [
        .executableTarget(
            name: "GemmaBench",
            dependencies: [
                .product(name: "GemmaCore", package: "GemmaCore"),
            ]
        ),
    ]
)
