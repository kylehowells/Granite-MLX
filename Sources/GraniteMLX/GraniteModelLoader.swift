import Foundation
import Dispatch
import Hub
import MLX

/// A validated Granite speech checkpoint loaded into MLX arrays.
public struct GraniteModelArtifact: @unchecked Sendable {
    /// Local checkpoint directory.
    public let directory: URL
    /// Parsed Granite architecture and quantization configuration.
    public let configuration: GraniteModelConfiguration
    /// Number of tensors retained after loading and normalization.
    public let tensorCount: Int
    let weights: [String: MLXArray]

    /// Location of the checkpoint's safetensors file.
    public var weightsURL: URL { directory.appendingPathComponent("model.safetensors") }
}

/// Loads and validates local or Hugging Face Granite speech checkpoints.
public enum GraniteModelLoader {
    /// Recommended speech checkpoint used when callers do not select one.
    public static let defaultModelID = "iky1e/granite-speech-5.0-470m-turboctc-mlx-q8"

    /// Loads either a local converted checkpoint directory or a Hugging Face
    /// repository ID. The model directory is cached by swift-transformers.
    public static func load(
        source: String = defaultModelID,
        hfToken: String? = nil,
        progressHandler: GraniteModelDownloadProgressHandler? = nil,
        cancellationToken: GraniteCancellationToken? = nil
    ) throws -> GraniteModelArtifact {
        try cancellationToken?.checkCancellation(operation: "Speech model loading")
        let localURL = URL(fileURLWithPath: source)
        if FileManager.default.fileExists(atPath: localURL.path) {
            return try load(from: localURL)
        }
        let directory = try GraniteModelCache.download(
            source, kind: .speech, hfToken: hfToken,
            cancellationToken: cancellationToken, progressHandler: progressHandler)
        try cancellationToken?.checkCancellation(operation: "Speech model loading")
        return try load(from: directory)
    }

    /// Loads and validates a converted Granite checkpoint directory.
    public static func load(from directory: URL) throws -> GraniteModelArtifact {
        let configURL = directory.appendingPathComponent("config.json")
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: configURL.path),
              FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw GraniteRecognizerError.invalidModel(directory)
        }
        let configuration: GraniteModelConfiguration
        let loadedWeights: [String: MLXArray]
        do {
            let configData = try Data(contentsOf: configURL)
            configuration = try JSONDecoder().decode(GraniteModelConfiguration.self, from: configData)
            loadedWeights = try MLX.loadArrays(url: weightsURL)
        } catch let error as GraniteDiagnosticError {
            throw error
        } catch {
            throw GraniteOperationError.underlying(
                code: "GMLX-RUNTIME-003", operation: "Speech model loading",
                details: "directory=\(directory.path); error=\(error)")
        }
        var weights: [String: MLXArray] = [:]
        weights.reserveCapacity(loadedWeights.count)
        for (key, value) in loadedWeights {
            if key.hasSuffix(".num_batches_tracked") { continue }
            if key.hasSuffix(".conv.depthwise_conv.weight"), value.ndim == 3,
               value.dim(1) == 1, value.dim(2) > 1 {
                weights[key] = value.transposed(0, 2, 1)
            } else {
                weights[key] = value
            }
        }
        let required = [
            "encoder.input_linear.weight",
            "encoder.input_linear.bias",
            "encoder.out.weight",
            "encoder.out.bias",
        ]
        let missing = required.filter { weights[$0] == nil }
        guard missing.isEmpty else {
            throw GraniteRecognizerError.unsupportedConfiguration("Converted Granite artifact is missing tensors: \(missing.joined(separator: ", "))")
        }
        if let quantization = configuration.quantization {
            try validateQuantizedWeights(weights, configuration: quantization)
        }
        return GraniteModelArtifact(
            directory: directory,
            configuration: configuration,
            tensorCount: weights.count,
            weights: weights
        )
    }

    static func validateQuantizedWeights(
        _ weights: [String: MLXArray],
        configuration: GraniteQuantizationConfiguration
    ) throws {
        try validateQuantizedWeightShapes(
            weights.mapValues(\.shape), configuration: configuration)
    }

    static func validateQuantizedWeightShapes(
        _ shapes: [String: [Int]],
        configuration: GraniteQuantizationConfiguration
    ) throws {
        guard configuration.mode == .affine else {
            throw GraniteRecognizerError.unsupportedConfiguration(
                "Granite-MLX currently supports affine quantized checkpoints, not \(configuration.mode.rawValue)."
            )
        }
        let configuredGroupSizes = [configuration.groupSize] + Array(configuration.groupSizes.values)
        guard [2, 3, 4, 5, 6, 8].contains(configuration.bits),
              configuredGroupSizes.allSatisfy({ [32, 64, 128].contains($0) }) else {
            throw GraniteRecognizerError.unsupportedConfiguration(
                "Unsupported affine quantization: \(configuration.bits)-bit, group size \(configuration.groupSize)."
            )
        }

        let packedWeights = shapes.filter { key, shape in
            key.hasSuffix(".weight") && shape.count == 2
        }
        guard !packedWeights.isEmpty else {
            throw GraniteRecognizerError.unsupportedConfiguration(
                "Quantized checkpoint contains no packed matrix weights."
            )
        }
        for (key, weightShape) in packedWeights {
            let base = String(key.dropLast(".weight".count))
            let groupSize = configuration.groupSize(for: base)
            guard let scalesShape = shapes["\(base).scales"],
                  let biasesShape = shapes["\(base).biases"] else {
                throw GraniteRecognizerError.unsupportedConfiguration(
                    "Quantized checkpoint is missing scales or biases for \(base)."
                )
            }
            guard scalesShape.count == 2, biasesShape == scalesShape,
                  weightShape[0] == scalesShape[0],
                  weightShape[1] * 32 / configuration.bits
                    == scalesShape[1] * groupSize else {
                throw GraniteRecognizerError.unsupportedConfiguration(
                    "Quantized tensor shapes are inconsistent for \(base)."
                )
            }
        }
    }

    private final class ResultBox<Value>: @unchecked Sendable {
        private let lock = NSLock()
        private var result: Result<Value, Error>?
        func set(_ result: Result<Value, Error>) { lock.lock(); self.result = result; lock.unlock() }
        func get() -> Result<Value, Error>? { lock.lock(); defer { lock.unlock() }; return result }
    }

    static func runBlocking<T>(
        cancellationToken: GraniteCancellationToken? = nil,
        _ operation: @escaping @Sendable () async throws -> T
    ) throws -> T {
        let box = ResultBox<T>()
        let semaphore = DispatchSemaphore(value: 0)
        let task = Task.detached(priority: .userInitiated) {
            do { box.set(.success(try await operation())) }
            catch { box.set(.failure(error)) }
            semaphore.signal()
        }
        while semaphore.wait(timeout: .now() + .milliseconds(100)) == .timedOut {
            if cancellationToken?.isCancelled == true { task.cancel() }
        }
        try cancellationToken?.checkCancellation(operation: "Model download")
        guard let result = box.get() else {
            throw GraniteOperationError.underlying(
                code: "GMLX-MODEL-008", operation: "Model download",
                details: "detached download task completed without a Result value")
        }
        return try result.get()
    }
}
