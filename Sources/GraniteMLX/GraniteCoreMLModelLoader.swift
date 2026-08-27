import Foundation

/// Speech-inference backend selectable by applications and the CLI.
public enum GraniteSpeechBackend: String, Codable, Sendable, CaseIterable {
    /// Native Swift inference using MLX.
    case mlx
    /// Fixed-shape ML Program inference using Core ML.
    case coreML = "coreml"
}

/// Quantization metadata stored with a converted Core ML Granite checkpoint.
public struct GraniteCoreMLQuantizationConfiguration: Codable, Sendable, Equatable {
    /// Weight-compression method, such as `uniform_palettization`.
    public let method: String
    /// Number of bits used for compressed weights.
    public let bits: Int
    /// Core ML compression granularity.
    public let granularity: String
    /// Number of output channels sharing each palette and scale.
    public let groupSize: Int
}

/// Machine-readable configuration for a published Core ML Granite checkpoint.
public struct GraniteCoreMLModelConfiguration: Codable, Sendable, Equatable {
    /// Core ML Granite model-type identifier.
    public let modelType: String
    /// Runtime backend identifier.
    public let backend: String
    /// Relative path to the `.mlpackage` inside the repository.
    public let modelPackage: String
    /// Source Hugging Face model identifier.
    public let sourceModel: String
    /// Exact source-model revision used for conversion.
    public let sourceRevision: String
    /// Minimum Apple deployment target encoded in the ML Program.
    public let minimumDeploymentTarget: String
    /// Fixed number of frontend frames accepted by the model.
    public let featureFrames: Int
    /// Audio duration represented by the fixed input.
    public let audioSeconds: Double
    /// Number of greedy CTC frame IDs returned by the model.
    public let outputFrames: Int
    /// Human-readable weight-precision label.
    public let weightPrecision: String
    /// Weight-compression configuration.
    public let quantization: GraniteCoreMLQuantizationConfiguration
    /// Compute-unit policy recommended by the publisher.
    public let recommendedComputeUnits: String
    /// Recommended central long-audio chunk duration.
    public let recommendedChunkDurationSeconds: Double
    /// Recommended context duration on each side of a chunk.
    public let recommendedChunkContextSeconds: Double
    /// SHA-256 of the Core ML package's primary weight blob.
    public let weightSha256: String
}

/// Validated files belonging to a downloaded or local Core ML Granite model.
public struct GraniteCoreMLModelArtifact: Sendable {
    /// Repository or local model directory.
    public let directory: URL
    /// Converted Core ML package.
    public let modelURL: URL
    /// Directory containing Granite tokenizer and architecture configuration.
    public let tokenizerURL: URL
    /// Parsed Core ML conversion metadata.
    public let configuration: GraniteCoreMLModelConfiguration
}

/// Downloads and validates local or Hugging Face Core ML Granite checkpoints.
public enum GraniteCoreMLModelLoader {
    /// Recommended Core ML checkpoint used when callers do not select one.
    public static let defaultModelID =
        "iky1e/granite-speech-5.0-470m-turboctc-coreml-q8"

    /// Loads a local repository directory or downloads a Hugging Face model.
    ///
    /// Download progress, cancellation, authentication, and cache placement use
    /// the same model-management APIs as the MLX speech and punctuation models.
    /// - Parameters:
    ///   - source: Local repository directory, catalog alias, or Hugging Face
    ///     repository ID. The published Core ML Q8 model is used when omitted.
    ///   - hfToken: Optional Hugging Face token for private or gated repositories.
    ///   - progressHandler: Receives model cache and byte-weighted download events.
    ///   - cancellationToken: Cooperatively cancels before, during, or after download.
    /// - Returns: Validated model package, tokenizer directory, and conversion metadata.
    /// - Throws: ``GraniteModelManagementError`` for resolution, download, or
    ///   cache failures; ``GraniteCoreMLRecognizerError`` for incompatible model
    ///   contents; ``GraniteOperationError`` when cancelled.
    public static func load(
        source: String = defaultModelID,
        hfToken: String? = nil,
        progressHandler: GraniteModelDownloadProgressHandler? = nil,
        cancellationToken: GraniteCancellationToken? = nil
    ) throws -> GraniteCoreMLModelArtifact {
        try cancellationToken?.checkCancellation(operation: "Core ML model loading")
        let localURL = URL(fileURLWithPath: source).standardizedFileURL
        if FileManager.default.fileExists(atPath: localURL.path) {
            return try load(from: localURL)
        }
        let directory = try GraniteModelCache.download(
            source, kind: .coreMLSpeech, hfToken: hfToken,
            cancellationToken: cancellationToken,
            progressHandler: progressHandler)
        try cancellationToken?.checkCancellation(operation: "Core ML model loading")
        return try load(from: directory)
    }

    /// Loads and validates a materialized Core ML model repository directory.
    /// - Parameter directory: Directory containing `coreml_config.json`, the
    ///   configured `.mlpackage`, and Granite tokenizer/configuration files.
    /// - Returns: A validated Core ML model artifact referencing local files.
    /// - Throws: ``GraniteCoreMLRecognizerError/invalidModel(_:details:)`` when
    ///   configuration, package, tokenizer, or cache state is missing or invalid.
    public static func load(from directory: URL) throws -> GraniteCoreMLModelArtifact {
        let directory = directory.standardizedFileURL
        let configurationURL = directory.appendingPathComponent("coreml_config.json")
        let configuration: GraniteCoreMLModelConfiguration
        do {
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            configuration = try decoder.decode(
                GraniteCoreMLModelConfiguration.self,
                from: Data(contentsOf: configurationURL))
        } catch {
            throw GraniteCoreMLRecognizerError.invalidModel(
                directory,
                details: "configuration=coreml_config.json; underlying=\(String(reflecting: error))")
        }
        guard configuration.modelType == "granite_speech5_coreml_ctc",
              configuration.backend == "coreml",
              let modelURL = GraniteModelCache.coreMLPackageURL(
                at: directory, relativePath: configuration.modelPackage),
              GraniteModelCache.state(at: directory, kind: .coreMLSpeech) == .downloaded else {
            throw GraniteCoreMLRecognizerError.invalidModel(
                directory,
                details: GraniteModelCache.cacheValidationDetails(
                    at: directory, kind: .coreMLSpeech))
        }
        return GraniteCoreMLModelArtifact(
            directory: directory,
            modelURL: modelURL,
            tokenizerURL: directory,
            configuration: configuration)
    }
}
