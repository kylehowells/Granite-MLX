import Foundation
import Hub

/// Functional role of a managed GraniteMLX checkpoint.
public enum GraniteManagedModelKind: String, Codable, Sendable {
    /// Granite speech-recognition model.
    case speech
    /// Granite speech-recognition model converted to a Core ML ML Program.
    case coreMLSpeech = "coreml-speech"
    /// Punctuation, capitalization, and sentence-boundary model.
    case punctuation
}

/// Metadata for a checkpoint published and supported by GraniteMLX.
public struct GranitePublishedModel: Codable, Sendable, Equatable {
    /// Short command-line alias such as `apache-q8`.
    public let alias: String
    /// Hugging Face `owner/repository` identifier.
    public let repositoryID: String
    /// Functional role of the model.
    public let kind: GraniteManagedModelKind
    /// Source-model family or license-oriented family label.
    public let family: String
    /// Weight precision or quantization label.
    public let precision: String
    /// Expected total materialized size, rounded from the published artifact.
    public let expectedBytes: Int64
    /// Whether this checkpoint is selected by default for its role.
    public let isDefault: Bool

    /// Creates published-model metadata.
    /// - Parameters:
    ///   - alias: Stable short name accepted by the CLI and library.
    ///   - repositoryID: Full Hugging Face `owner/repository` identifier.
    ///   - kind: Functional checkpoint role.
    ///   - family: Human-readable source/license family label.
    ///   - precision: Human-readable weight precision or quantization label.
    ///   - expectedBytes: Approximate materialized repository size in bytes.
    ///   - isDefault: Whether this checkpoint is recommended for its model role.
    public init(
        alias: String,
        repositoryID: String,
        kind: GraniteManagedModelKind,
        family: String,
        precision: String,
        expectedBytes: Int64,
        isDefault: Bool = false
    ) {
        self.alias = alias
        self.repositoryID = repositoryID
        self.kind = kind
        self.family = family
        self.precision = precision
        self.expectedBytes = expectedBytes
        self.isDefault = isDefault
    }
}

/// Stages emitted while checking or downloading model files.
public enum GraniteModelDownloadPhase: String, Codable, Sendable {
    /// Repository metadata and local state are being checked.
    case checking
    /// Repository files are being downloaded or resumed.
    case downloading
    /// A complete local checkpoint was found and reused.
    case cacheHit = "cache_hit"
    /// Download and validation completed successfully.
    case complete
}

/// Progress information for model download and cache operations.
public struct GraniteModelDownloadProgress: Sendable {
    /// Hugging Face repository identifier.
    public let repositoryID: String
    /// Functional role of the model.
    public let kind: GraniteManagedModelKind
    /// Materialized local checkpoint directory.
    public let cacheDirectory: URL
    /// Current download stage.
    public let phase: GraniteModelDownloadPhase
    /// Completion in the inclusive range `0...1`.
    public let fractionCompleted: Double
    /// Approximate final cache size when known.
    public let estimatedTotalBytes: Int64?
    /// Current transfer throughput when supplied by the Hub client.
    public let bytesPerSecond: Double?

    /// Creates a model-download progress value.
    /// - Parameters:
    ///   - repositoryID: Full Hugging Face repository identifier.
    ///   - kind: Functional checkpoint role being acquired.
    ///   - cacheDirectory: Destination materialized repository directory.
    ///   - phase: Current cache or download stage.
    ///   - fractionCompleted: Estimated completion in `0...1`.
    ///   - estimatedTotalBytes: Approximate final size, or `nil` when unknown.
    ///   - bytesPerSecond: Current transfer rate, or `nil` when unavailable.
    public init(
        repositoryID: String,
        kind: GraniteManagedModelKind,
        cacheDirectory: URL,
        phase: GraniteModelDownloadPhase,
        fractionCompleted: Double,
        estimatedTotalBytes: Int64?,
        bytesPerSecond: Double?
    ) {
        self.repositoryID = repositoryID
        self.kind = kind
        self.cacheDirectory = cacheDirectory
        self.phase = phase
        self.fractionCompleted = fractionCompleted
        self.estimatedTotalBytes = estimatedTotalBytes
        self.bytesPerSecond = bytesPerSecond
    }
}

/// Receives model download and cache progress updates.
public typealias GraniteModelDownloadProgressHandler = @Sendable (GraniteModelDownloadProgress) -> Void

/// On-disk state of a catalog checkpoint.
public enum GraniteModelCacheState: String, Codable, Sendable {
    /// No materialized files are present.
    case absent
    /// Some files exist, but the checkpoint is incomplete or invalid.
    case partial
    /// All required files and recognized configuration are present.
    case downloaded
}

/// A Granite-compatible checkpoint found in the local Swift Hub cache.
public struct GraniteCachedModel: Codable, Sendable {
    /// Hugging Face repository identifier.
    public let repositoryID: String
    /// Functional role of the checkpoint.
    public let kind: GraniteManagedModelKind
    /// Materialized checkpoint directory.
    public let directory: URL
    /// Logical size of files beneath ``directory``.
    public let sizeBytes: Int64
    /// Logical size of the compiled Core ML cache associated with this model.
    public let compiledCacheBytes: Int64
    /// Matching catalog alias, if this is a published GraniteMLX checkpoint.
    public let catalogAlias: String?
    /// Completeness state of the materialized checkpoint.
    public let state: GraniteModelCacheState
    /// Technical explanation when ``state`` is ``GraniteModelCacheState/partial``.
    public let stateDetails: String?
}

/// Errors produced by model catalog, download, validation, and cache operations.
public enum GraniteModelManagementError: Error, GraniteDiagnosticError {
    /// Repository ID does not have the required `owner/repository` form. The
    /// associated string is the rejected identifier.
    case invalidRepositoryID(String)
    /// Alias or ID cannot be resolved. The associated string is the rejected value.
    case unknownModel(String)
    /// No local files are present for the requested model. The associated string
    /// is its canonical repository identifier.
    case notDownloaded(String)
    /// Available disk capacity is too low for the expected checkpoint.
    /// - Parameters:
    ///   - required: Estimated bytes required to complete materialization.
    ///   - available: Bytes currently available on the destination volume.
    case insufficientDiskSpace(required: Int64, available: Int64)
    /// Hub download or repository lookup failed.
    /// - Parameters:
    ///   - repositoryID: Canonical repository being downloaded.
    ///   - cacheDirectory: Destination retaining any resumable partial files.
    ///   - details: Underlying Hub or network diagnostics.
    case downloadFailed(repositoryID: String, cacheDirectory: URL, details: String)
    /// Download returned without producing a complete compatible checkpoint.
    /// - Parameters:
    ///   - repositoryID: Canonical downloaded repository.
    ///   - cacheDirectory: Materialized directory that failed validation.
    ///   - details: Missing-file or compatibility validation details.
    case incompleteModel(repositoryID: String, cacheDirectory: URL, details: String)
    /// Removing a materialized checkpoint failed.
    /// - Parameters:
    ///   - repositoryID: Canonical repository selected for removal.
    ///   - cacheDirectory: Exact materialized directory being removed.
    ///   - details: Underlying filesystem or Core ML cache-removal diagnostics.
    case removalFailed(repositoryID: String, cacheDirectory: URL, details: String)

    /// Stable diagnostic identifier for the failure.
    public var diagnosticCode: String {
        switch self {
        case .invalidRepositoryID: "GMLX-MODEL-001"
        case .unknownModel: "GMLX-MODEL-002"
        case .notDownloaded: "GMLX-MODEL-003"
        case .insufficientDiskSpace: "GMLX-MODEL-004"
        case .downloadFailed: "GMLX-MODEL-005"
        case .incompleteModel: "GMLX-MODEL-006"
        case .removalFailed: "GMLX-MODEL-007"
        }
    }

    /// Low-level context useful for diagnostics.
    public var technicalDetails: String? {
        switch self {
        case .invalidRepositoryID(let value), .unknownModel(let value), .notDownloaded(let value):
            "model=\(value)"
        case .insufficientDiskSpace(let required, let available):
            "required_bytes=\(required); available_bytes=\(available)"
        case .downloadFailed(let id, let directory, let details),
             .incompleteModel(let id, let directory, let details),
             .removalFailed(let id, let directory, let details):
            "repository=\(id); cache=\(directory.path); underlying=\(details)"
        }
    }

    /// User-facing localized failure description containing the diagnostic code.
    public var errorDescription: String? {
        switch self {
        case .invalidRepositoryID(let value):
            "[\(diagnosticCode)] Invalid Hugging Face repository ID `\(value)`; expected `owner/repository`. Technical details: \(technicalDetails!)."
        case .unknownModel(let value):
            "[\(diagnosticCode)] Unknown model alias or repository ID `\(value)`. Run `granite-mlx models list`. Technical details: \(technicalDetails!)."
        case .notDownloaded(let value):
            "[\(diagnosticCode)] Model `\(value)` has no local cache files to remove. Technical details: \(technicalDetails!)."
        case .insufficientDiskSpace:
            "[\(diagnosticCode)] There is not enough free disk space for this model. Remove an unused model with `granite-mlx models remove`, or free disk space. Technical details: \(technicalDetails!)."
        case .downloadFailed(let id, _, _):
            "[\(diagnosticCode)] Could not download `\(id)`. Check the network connection, repository ID, and authentication. Partial files are retained so a later download can resume. Technical details: \(technicalDetails!)."
        case .incompleteModel(let id, _, _):
            "[\(diagnosticCode)] `\(id)` is incomplete or incompatible after download. Run `granite-mlx models download \(id)` to repair it, or remove it first. Technical details: \(technicalDetails!)."
        case .removalFailed(let id, _, _):
            "[\(diagnosticCode)] Could not remove cached model `\(id)`. Check file permissions and whether another process is using it. Technical details: \(technicalDetails!)."
        }
    }
}

/// Catalog of GraniteMLX speech and punctuation checkpoints.
public enum GraniteModelCatalog {
    /// All officially published GraniteMLX checkpoints.
    public static let models: [GranitePublishedModel] = [
        speech("apache-fp16", "granite-speech-5.0-470m-turboctc-mlx-fp16", "Apache 2.0", "FP16", 947_220_480),
        speech("apache-q8", "granite-speech-5.0-470m-turboctc-mlx-q8", "Apache 2.0", "Q8", 489_840_640, isDefault: true),
        speech("apache-q6", "granite-speech-5.0-470m-turboctc-mlx-q6", "Apache 2.0", "Q6", 386_539_520),
        speech("apache-q5", "granite-speech-5.0-470m-turboctc-mlx-q5", "Apache 2.0", "Q5", 327_516_160),
        speech("apache-q4", "granite-speech-5.0-470m-turboctc-mlx-q4", "Apache 2.0", "Q4", 268_488_704),
        speech("nc-fp16", "granite-speech-5.0-470m-turboctc-nc-mlx-fp16", "Non-commercial", "FP16", 948_297_728),
        speech("nc-q8", "granite-speech-5.0-470m-turboctc-nc-mlx-q8", "Non-commercial", "Q8", 490_917_888),
        speech("nc-q6", "granite-speech-5.0-470m-turboctc-nc-mlx-q6", "Non-commercial", "Q6", 387_616_768),
        speech("nc-q5", "granite-speech-5.0-470m-turboctc-nc-mlx-q5", "Non-commercial", "Q5", 328_593_408),
        speech("nc-q4", "granite-speech-5.0-470m-turboctc-nc-mlx-q4", "Non-commercial", "Q4", 269_565_952),
        coreMLSpeech(
            "apache-coreml-q8", "granite-speech-5.0-470m-turboctc-coreml-q8",
            "Apache 2.0", "Core ML Q8", 693_040_765, isDefault: true),
        punctuation("punctuation-fp16", "FP16", 107_429_888),
        punctuation("punctuation-q8", "Q8", 58_527_744, isDefault: true),
        punctuation("punctuation-q6", "Q6", 45_486_080),
        punctuation("punctuation-q5", "Q5", 38_965_248),
        punctuation("punctuation-q4", "Q4", 32_440_320),
    ]

    /// Resolves a catalog alias or validates a custom Hugging Face repository ID.
    /// - Parameter aliasOrID: Published alias or `owner/repository` identifier.
    /// - Returns: Canonical repository ID and catalog metadata when known.
    /// - Throws: ``GraniteModelManagementError/unknownModel(_:)`` when the value
    ///   is neither a catalog entry nor a valid repository ID.
    public static func resolve(_ aliasOrID: String) throws -> (id: String, model: GranitePublishedModel?) {
        if let model = models.first(where: {
            $0.alias.caseInsensitiveCompare(aliasOrID) == .orderedSame
                || $0.repositoryID.caseInsensitiveCompare(aliasOrID) == .orderedSame
        }) {
            return (model.repositoryID, model)
        }
        guard GraniteModelCache.isValidRepositoryID(aliasOrID) else {
            throw GraniteModelManagementError.unknownModel(aliasOrID)
        }
        return (aliasOrID, nil)
    }

    /// Returns catalog metadata matching a full repository ID.
    /// - Parameter repositoryID: Case-insensitive full Hugging Face identifier.
    /// - Returns: Matching catalog entry, or `nil` for custom repositories.
    public static func model(for repositoryID: String) -> GranitePublishedModel? {
        models.first { $0.repositoryID.caseInsensitiveCompare(repositoryID) == .orderedSame }
    }

    private static func speech(
        _ alias: String, _ repository: String, _ family: String, _ precision: String,
        _ bytes: Int64, isDefault: Bool = false
    ) -> GranitePublishedModel {
        GranitePublishedModel(
            alias: alias, repositoryID: "iky1e/\(repository)", kind: .speech,
            family: family, precision: precision, expectedBytes: bytes,
            isDefault: isDefault)
    }

    private static func punctuation(
        _ alias: String, _ precision: String, _ bytes: Int64, isDefault: Bool = false
    ) -> GranitePublishedModel {
        GranitePublishedModel(
            alias: alias,
            repositoryID: "iky1e/punctuation-fullstop-truecase-english-mlx-\(precision.lowercased())",
            kind: .punctuation, family: "Formatter", precision: precision,
            expectedBytes: bytes, isDefault: isDefault)
    }

    private static func coreMLSpeech(
        _ alias: String, _ repository: String, _ family: String, _ precision: String,
        _ bytes: Int64, isDefault: Bool = false
    ) -> GranitePublishedModel {
        GranitePublishedModel(
            alias: alias, repositoryID: "iky1e/\(repository)", kind: .coreMLSpeech,
            family: family, precision: precision, expectedBytes: bytes,
            isDefault: isDefault)
    }
}

/// Inspects, downloads, validates, and removes GraniteMLX model cache entries.
public enum GraniteModelCache {
    private static var hubDirectory: URL? {
        ProcessInfo.processInfo.environment["GRANITE_MLX_HUB_DIRECTORY"].map {
            URL(fileURLWithPath: $0).standardizedFileURL
        }
    }

    private static func hub(hfToken: String? = nil) -> HubApi {
        HubApi(downloadBase: hubDirectory, hfToken: hfToken)
    }

    /// Root of the Swift Hub materialized model cache.
    ///
    /// Set `GRANITE_MLX_HUB_DIRECTORY` to override the parent Hugging Face
    /// directory. This is useful for isolated application and test caches.
    public static var rootDirectory: URL {
        let hub = hub()
        return hub.localRepoLocation(.init(id: "placeholder/repository"))
            .deletingLastPathComponent().deletingLastPathComponent()
    }

    /// Returns the materialized cache directory for a Hugging Face repository ID.
    /// - Parameter repositoryID: Full `owner/repository` identifier.
    /// - Returns: Standardized local Hub materialization directory. The directory
    ///   is not created by this lookup.
    /// - Throws: ``GraniteModelManagementError/invalidRepositoryID(_:)`` when the
    ///   identifier does not have a safe `owner/repository` form.
    public static func directory(for repositoryID: String) throws -> URL {
        guard isValidRepositoryID(repositoryID) else {
            throw GraniteModelManagementError.invalidRepositoryID(repositoryID)
        }
        return hub().localRepoLocation(.init(id: repositoryID)).standardizedFileURL
    }

    /// Returns the current completeness state for a model cache entry.
    /// - Parameters:
    ///   - repositoryID: Full Hugging Face repository identifier.
    ///   - kind: Expected checkpoint role, or `nil` to accept any recognized role.
    /// - Returns: ``GraniteModelCacheState/absent``,
    ///   ``GraniteModelCacheState/partial``, or
    ///   ``GraniteModelCacheState/downloaded`` after local validation.
    public static func state(
        of repositoryID: String,
        kind: GraniteManagedModelKind? = nil
    ) -> GraniteModelCacheState {
        guard let directory = try? directory(for: repositoryID) else { return .absent }
        return state(at: directory, kind: kind)
    }

    static func state(
        at directory: URL,
        kind: GraniteManagedModelKind? = nil
    ) -> GraniteModelCacheState {
        guard FileManager.default.fileExists(atPath: directory.path) else { return .absent }
        return detectedKind(at: directory).map { detected in
            kind == nil || detected == kind ? .downloaded : .partial
        } ?? .partial
    }

    /// Indicates whether a complete compatible checkpoint is cached.
    /// - Parameters:
    ///   - repositoryID: Full Hugging Face repository identifier.
    ///   - kind: Expected checkpoint role, or `nil` for any recognized role.
    /// - Returns: `true` only when all required files and configuration validate.
    public static func isDownloaded(_ repositoryID: String, kind: GraniteManagedModelKind? = nil) -> Bool {
        state(of: repositoryID, kind: kind) == .downloaded
    }

    /// Lists complete and partial catalog checkpoints plus compatible custom checkpoints.
    /// - Returns: Cached model records sorted case-insensitively by repository ID.
    public static func downloadedModels() -> [GraniteCachedModel] {
        let manager = FileManager.default
        guard let owners = try? manager.contentsOfDirectory(
            at: rootDirectory, includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]) else { return [] }
        var result: [GraniteCachedModel] = []
        for owner in owners where isDirectory(owner) {
            guard let repositories = try? manager.contentsOfDirectory(
                at: owner, includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]) else { continue }
            for repository in repositories where isDirectory(repository) {
                guard let kind = detectedKind(at: repository) else { continue }
                let repositoryID = "\(owner.lastPathComponent)/\(repository.lastPathComponent)"
                result.append(GraniteCachedModel(
                    repositoryID: repositoryID,
                    kind: kind,
                    directory: repository,
                    sizeBytes: directorySize(repository),
                    compiledCacheBytes: compiledCacheSize(
                        at: repository, kind: kind),
                    catalogAlias: GraniteModelCatalog.model(for: repositoryID)?.alias,
                    state: .downloaded,
                    stateDetails: nil))
            }
        }
        let existingIDs = Set(result.map { $0.repositoryID.lowercased() })
        for model in GraniteModelCatalog.models where !existingIDs.contains(model.repositoryID.lowercased()) {
            guard let directory = try? directory(for: model.repositoryID),
                  FileManager.default.fileExists(atPath: directory.path) else { continue }
            result.append(GraniteCachedModel(
                repositoryID: model.repositoryID, kind: model.kind,
                directory: directory, sizeBytes: directorySize(directory),
                compiledCacheBytes: compiledCacheSize(
                    at: directory, kind: model.kind),
                catalogAlias: model.alias, state: .partial,
                stateDetails: cacheValidationDetails(at: directory, kind: model.kind)))
        }
        return result.sorted { $0.repositoryID.localizedCaseInsensitiveCompare($1.repositoryID) == .orderedAscending }
    }

    /// Downloads or resumes a checkpoint and validates the materialized result.
    ///
    /// - Parameters:
    ///   - aliasOrID: Catalog alias or Hugging Face repository ID.
    ///   - requestedKind: Explicit model role for custom repositories.
    ///   - hfToken: Optional Hugging Face token. When `nil`, Hub environment-token resolution applies.
    ///   - cancellationToken: Optional cooperative cancellation token.
    ///   - progressHandler: Optional download progress callback.
    /// - Returns: Validated materialized checkpoint directory.
    /// - Throws: ``GraniteModelManagementError`` for invalid IDs, insufficient
    ///   disk space, transfer failures, or incomplete artifacts, and
    ///   ``GraniteOperationError`` when cancellation is requested.
    @discardableResult
    public static func download(
        _ aliasOrID: String,
        kind requestedKind: GraniteManagedModelKind? = nil,
        hfToken: String? = nil,
        cancellationToken: GraniteCancellationToken? = nil,
        progressHandler: GraniteModelDownloadProgressHandler? = nil
    ) throws -> URL {
        let resolved = try GraniteModelCatalog.resolve(aliasOrID)
        let id = resolved.id
        let kind = requestedKind ?? resolved.model?.kind ?? inferredKind(from: id)
        let expectedBytes = resolved.model?.expectedBytes
        let destination = try directory(for: id)
        let event: @Sendable (GraniteModelDownloadPhase, Double, Double?) -> Void = { phase, fraction, speed in
            progressHandler?(GraniteModelDownloadProgress(
                repositoryID: id, kind: kind, cacheDirectory: destination,
                phase: phase, fractionCompleted: fraction,
                estimatedTotalBytes: expectedBytes, bytesPerSecond: speed))
        }
        try cancellationToken?.checkCancellation(operation: "Model download")
        if isDownloaded(id, kind: kind) {
            event(.cacheHit, 1, nil)
            return destination
        }
        event(.checking, 0, nil)
        try preflightDiskSpace(at: destination, expectedBytes: expectedBytes)
        let hub = hub(hfToken: hfToken)
        let patterns: [String] = switch kind {
        case .punctuation:
            ["*.safetensors", "*.json", "*.model", "*.yaml"]
        case .speech:
            ["*.safetensors", "*.json", "*.txt", "*.model"]
        case .coreMLSpeech:
            ["*.mlpackage/*", "*.json", "*.txt", "*.model"]
        }
        let downloaded: URL
        do {
            downloaded = try GraniteModelLoader.runBlocking(cancellationToken: cancellationToken) {
                let metadata = try? await hub.getFileMetadata(
                    from: .init(id: id), matching: patterns)
                let fileSizes: [Int64]? = metadata.flatMap { values in
                    let sizes = values.compactMap(\.size).map(Int64.init)
                    return sizes.count == values.count ? sizes : nil
                }
                return try await hub.snapshot(from: id, matching: patterns) { progress, speed in
                    event(
                        .downloading,
                        weightedDownloadFraction(
                            progress.fractionCompleted, fileSizes: fileSizes),
                        speed)
                }
            }
        } catch let error as GraniteOperationError { throw error }
        catch {
            throw GraniteModelManagementError.downloadFailed(
                repositoryID: id, cacheDirectory: destination,
                details: String(reflecting: error))
        }
        try cancellationToken?.checkCancellation(operation: "Model download")
        guard detectedKind(at: downloaded) == kind else {
            throw GraniteModelManagementError.incompleteModel(
                repositoryID: id, cacheDirectory: downloaded,
                details: cacheValidationDetails(at: downloaded, kind: kind))
        }
        event(.complete, 1, nil)
        return downloaded
    }

    /// Permanently removes one exact materialized Hub repository directory.
    /// It can be restored by downloading the repository again.
    /// - Parameter aliasOrID: Catalog alias or exact repository ID to remove.
    /// - Returns: Metadata and reclaimed-size information captured before removal.
    /// - Throws: ``GraniteModelManagementError`` when resolution fails, no cache
    ///   exists, or the model and associated compiled Core ML cache cannot be removed.
    @discardableResult
    public static func remove(_ aliasOrID: String) throws -> GraniteCachedModel {
        let resolved = try GraniteModelCatalog.resolve(aliasOrID)
        let destination = try directory(for: resolved.id)
        guard FileManager.default.fileExists(atPath: destination.path) else {
            throw GraniteModelManagementError.notDownloaded(resolved.id)
        }
        let kind = detectedKind(at: destination) ?? resolved.model?.kind ?? inferredKind(from: resolved.id)
        let cacheState = state(of: resolved.id, kind: kind)
        let record = GraniteCachedModel(
            repositoryID: resolved.id, kind: kind, directory: destination,
            sizeBytes: directorySize(destination),
            compiledCacheBytes: compiledCacheSize(at: destination, kind: kind),
            catalogAlias: resolved.model?.alias,
            state: cacheState,
            stateDetails: cacheState == .partial ? cacheValidationDetails(at: destination, kind: kind) : nil)
        do {
            if kind == .coreMLSpeech,
               let package = coreMLPackageURL(at: destination) {
                try GraniteCoreMLRecognizer.removeCompiledModelCache(for: package)
            }
            try FileManager.default.removeItem(at: destination)
        }
        catch {
            throw GraniteModelManagementError.removalFailed(
                repositoryID: resolved.id, cacheDirectory: destination,
                details: String(reflecting: error))
        }
        return record
    }

    /// Recursively computes logical file size without following symbolic links.
    /// - Parameter directory: Directory tree whose regular files should be measured.
    /// - Returns: Sum of regular-file logical sizes in bytes. Unreadable entries
    ///   and symbolic links are ignored.
    public static func directorySize(_ directory: URL) -> Int64 {
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .fileSizeKey, .isSymbolicLinkKey]
        guard let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: Array(keys)) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            guard let values = try? file.resourceValues(forKeys: keys),
                  values.isRegularFile == true, values.isSymbolicLink != true else { continue }
            total += Int64(values.fileSize ?? 0)
        }
        return total
    }

    static func isValidRepositoryID(_ value: String) -> Bool {
        let parts = value.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2 else { return false }
        return parts.allSatisfy { part in
            !part.isEmpty && part != "." && part != ".."
                && !part.contains("\\") && !part.contains(":")
        }
    }

    static func detectedKind(at directory: URL) -> GraniteManagedModelKind? {
        let manager = FileManager.default
        if manager.fileExists(
            atPath: directory.appendingPathComponent("coreml_config.json").path),
           cacheValidationDetails(at: directory, kind: .coreMLSpeech)
            == "required files and configuration are valid" {
            return .coreMLSpeech
        }
        let weights = directory.appendingPathComponent("model.safetensors")
        guard manager.fileExists(atPath: weights.path) else { return nil }
        let tokenizer = directory.appendingPathComponent("tokenizer.json")
        guard manager.fileExists(atPath: tokenizer.path) else { return nil }
        let kind: GraniteManagedModelKind?
        let punctuationConfig = directory.appendingPathComponent("mlx_config.json")
        if let data = try? Data(contentsOf: punctuationConfig),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           object["architecture"] as? String == "bert-punctuation-capitalization-segmentation" {
            kind = .punctuation
        } else {
            let speechConfig = directory.appendingPathComponent("config.json")
            if let data = try? Data(contentsOf: speechConfig),
               let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               object["model_type"] as? String == "granite_speech5_ctc" {
                kind = .speech
            } else {
                kind = nil
            }
        }
        guard let kind, safetensorsValidationDetails(at: weights, kind: kind) == nil else {
            return nil
        }
        return kind
    }

    static func cacheValidationDetails(
        at directory: URL, kind: GraniteManagedModelKind
    ) -> String {
        if kind == .coreMLSpeech {
            let required = ["coreml_config.json", "config.json", "tokenizer.json"]
            let missing = required.filter {
                !FileManager.default.fileExists(
                    atPath: directory.appendingPathComponent($0).path)
            }
            if !missing.isEmpty {
                return "missing_files=\(missing.joined(separator: ","))"
            }
            let configurationURL = directory.appendingPathComponent(
                "coreml_config.json")
            let configuration: GraniteCoreMLModelConfiguration
            do {
                let decoder = JSONDecoder()
                decoder.keyDecodingStrategy = .convertFromSnakeCase
                configuration = try decoder.decode(
                    GraniteCoreMLModelConfiguration.self,
                    from: Data(contentsOf: configurationURL))
            } catch {
                return "invalid_coreml=configuration_decode_failed; error=\(String(reflecting: error))"
            }
            guard configuration.modelType == "granite_speech5_coreml_ctc",
                  configuration.backend == "coreml",
                  configuration.featureFrames > 0,
                  configuration.outputFrames > 0,
                  configuration.audioSeconds > 0,
                  configuration.quantization.bits > 0,
                  configuration.weightSha256.count == 64,
                  configuration.weightSha256.allSatisfy({ $0.isHexDigit }) else {
                return "invalid_coreml=configuration_values; model_type=\(configuration.modelType); backend=\(configuration.backend); feature_frames=\(configuration.featureFrames); output_frames=\(configuration.outputFrames); weight_sha256_length=\(configuration.weightSha256.count)"
            }
            guard let package = coreMLPackageURL(at: directory) else {
                return "invalid_coreml=model_package is absent, unsafe, or not an mlpackage"
            }
            let packageFiles = [
                "Manifest.json",
                "Data/com.apple.CoreML/model.mlmodel",
                "Data/com.apple.CoreML/weights/weight.bin",
            ]
            let missingPackageFiles = packageFiles.filter {
                !FileManager.default.fileExists(
                    atPath: package.appendingPathComponent($0).path)
            }
            if !missingPackageFiles.isEmpty {
                return "invalid_coreml=missing_package_files; files=\(missingPackageFiles.joined(separator: ","))"
            }
            let weight = package.appendingPathComponent(
                "Data/com.apple.CoreML/weights/weight.bin")
            let weightBytes = (try? weight.resourceValues(
                forKeys: [.fileSizeKey]).fileSize) ?? 0
            guard weightBytes > 100 * 1_024 * 1_024 else {
                return "invalid_coreml=weight_blob_too_small; bytes=\(weightBytes)"
            }
            return "required files and configuration are valid"
        }
        let required = kind == .speech
            ? ["model.safetensors", "config.json", "tokenizer.json"]
            : ["model.safetensors", "mlx_config.json", "tokenizer.json"]
        let missing = required.filter {
            !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
        if !missing.isEmpty { return "missing_files=\(missing.joined(separator: ","))" }
        let weights = directory.appendingPathComponent("model.safetensors")
        if let details = safetensorsValidationDetails(at: weights, kind: kind) {
            return details
        }
        return "required files exist but configuration architecture or model kind is invalid"
    }

    static func coreMLPackageURL(
        at directory: URL, relativePath: String? = nil
    ) -> URL? {
        let directory = directory.standardizedFileURL
        let configuredPath: String
        if let relativePath {
            configuredPath = relativePath
        } else {
            let configurationURL = directory.appendingPathComponent(
                "coreml_config.json")
            guard let data = try? Data(contentsOf: configurationURL),
                  let object = try? JSONSerialization.jsonObject(with: data)
                    as? [String: Any],
                  let value = object["model_package"] as? String else {
                return nil
            }
            configuredPath = value
        }
        guard !configuredPath.hasPrefix("/"),
              !configuredPath.split(separator: "/").contains("..") else {
            return nil
        }
        let package = directory.appendingPathComponent(configuredPath)
            .standardizedFileURL
        guard package.path.hasPrefix(directory.path + "/"),
              package.pathExtension == "mlpackage",
              (try? package.resourceValues(
                forKeys: [.isDirectoryKey]).isDirectory) == true else {
            return nil
        }
        return package
    }

    static func safetensorsValidationDetails(
        at weights: URL, kind: GraniteManagedModelKind
    ) -> String? {
        do {
            let handle = try FileHandle(forReadingFrom: weights)
            defer { try? handle.close() }
            guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
                return "invalid_safetensors=missing 8-byte header length"
            }
            var headerLength: UInt64 = 0
            for (offset, byte) in prefix.enumerated() {
                headerLength |= UInt64(byte) << UInt64(offset * 8)
            }
            let attributes = try FileManager.default.attributesOfItem(atPath: weights.path)
            let fileSize = (attributes[.size] as? NSNumber)?.uint64Value ?? 0
            guard headerLength > 0, headerLength <= 64 * 1_024 * 1_024,
                  headerLength <= fileSize.saturatingSubtract(8) else {
                return "invalid_safetensors=header_length_\(headerLength); file_bytes=\(fileSize)"
            }
            guard let header = try handle.read(upToCount: Int(headerLength)),
                  header.count == Int(headerLength),
                  let object = try JSONSerialization.jsonObject(with: header) as? [String: Any] else {
                return "invalid_safetensors=header is truncated or is not a JSON object"
            }
            let required = kind == .speech
                ? ["encoder.input_linear.weight", "encoder.out.weight"]
                : ["embeddings.word.weight", "decoder.post.1.weight"]
            let missing = required.filter { object[$0] == nil }
            return missing.isEmpty
                ? nil
                : "invalid_safetensors=missing_required_tensors; tensors=\(missing.joined(separator: ","))"
        } catch {
            return "invalid_safetensors=unreadable; error=\(String(reflecting: error))"
        }
    }

    private static func inferredKind(from repositoryID: String) -> GraniteManagedModelKind {
        if repositoryID.localizedCaseInsensitiveContains("punctuation") {
            return .punctuation
        }
        if repositoryID.localizedCaseInsensitiveContains("coreml") {
            return .coreMLSpeech
        }
        return .speech
    }

    private static func compiledCacheSize(
        at directory: URL, kind: GraniteManagedModelKind
    ) -> Int64 {
        guard kind == .coreMLSpeech,
              let package = coreMLPackageURL(at: directory) else { return 0 }
        return GraniteCoreMLRecognizer.compiledModelCacheSize(for: package)
    }

    static func weightedDownloadFraction(
        _ fileWeightedFraction: Double, fileSizes: [Int64]?
    ) -> Double {
        guard let fileSizes, !fileSizes.isEmpty else {
            return fileWeightedFraction
        }
        let total = fileSizes.reduce(Int64(0), +)
        guard total > 0 else { return fileWeightedFraction }
        let clamped = min(1, max(0, fileWeightedFraction))
        if clamped >= 1 { return 1 }
        let scaled = clamped * Double(fileSizes.count)
        let activeIndex = min(fileSizes.count - 1, Int(scaled))
        let activeFraction = scaled - Double(activeIndex)
        let completed = fileSizes[..<activeIndex].reduce(Int64(0), +)
        let active = Double(fileSizes[activeIndex]) * activeFraction
        return min(1, (Double(completed) + active) / Double(total))
    }

    private static func isDirectory(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
    }

    private static func preflightDiskSpace(at destination: URL, expectedBytes: Int64?) throws {
        guard let expectedBytes else { return }
        let parent = rootDirectory.deletingLastPathComponent()
        let values = try? parent.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        guard let available = values?.volumeAvailableCapacityForImportantUsage else { return }
        let reserve: Int64 = 100 * 1_024 * 1_024
        guard available >= expectedBytes + reserve else {
            throw GraniteModelManagementError.insufficientDiskSpace(
                required: expectedBytes + reserve, available: available)
        }
    }
}

private extension UInt64 {
    func saturatingSubtract(_ value: UInt64) -> UInt64 {
        self >= value ? self - value : 0
    }
}
