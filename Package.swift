// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Granite-MLX",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "GraniteMLX", targets: ["GraniteMLX"]),
        .executable(name: "granite-mlx", targets: ["GraniteMLXCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMajor(from: "0.30.6")),
        .package(url: "https://github.com/huggingface/swift-transformers.git", .upToNextMajor(from: "1.1.6")),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "GraniteMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "GraniteMLXCLI",
            dependencies: [
                "GraniteMLX",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(name: "GraniteMLXTests", dependencies: ["GraniteMLX"]),
    ]
)
