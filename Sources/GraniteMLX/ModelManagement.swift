import Foundation
import Hub

public enum GraniteManagedModelKind: String, Codable, Sendable {
    case speech
    case punctuation
}

public struct GranitePublishedModel: Codable, Sendable, Equatable {
    public let alias: String
    public let repositoryID: String
    public let kind: GraniteManagedModelKind
    public let family: String
    public let precision: String
    /// Expected total materialized size, rounded from the published artifact.
    public let expectedBytes: Int64
    public let isDefault: Bool

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

public enum GraniteModelDownloadPhase: String, Codable, Sendable {
    case checking
    case downloading
    case cacheHit = "cache_hit"
    case complete
}

public struct GraniteModelDownloadProgress: Sendable {
    public let repositoryID: String
    public let kind: GraniteManagedModelKind
    public let cacheDirectory: URL
    public let phase: GraniteModelDownloadPhase
    public let fractionCompleted: Double
    public let estimatedTotalBytes: Int64?
    public let bytesPerSecond: Double?

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

public typealias GraniteModelDownloadProgressHandler = @Sendable (GraniteModelDownloadProgress) -> Void

public struct GraniteCachedModel: Codable, Sendable {
    public let repositoryID: String
    public let kind: GraniteManagedModelKind
    public let directory: URL
    public let sizeBytes: Int64
    public let catalogAlias: String?
}

public enum GraniteModelManagementError: Error, LocalizedError {
    case invalidRepositoryID(String)
    case unknownModel(String)
    case notDownloaded(String)
    case insufficientDiskSpace(required: Int64, available: Int64)

    public var errorDescription: String? {
        switch self {
        case .invalidRepositoryID(let value):
            "Invalid Hugging Face repository ID: \(value). Expected owner/repository."
        case .unknownModel(let value):
            "Unknown model alias or repository ID: \(value). Run `granite-mlx models list`."
        case .notDownloaded(let value):
            "Model is not downloaded: \(value)"
        case .insufficientDiskSpace(let required, let available):
            "Insufficient disk space. Approximately \(required) bytes are required; \(available) bytes are available."
        }
    }
}

public enum GraniteModelCatalog {
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
        punctuation("punctuation-fp16", "FP16", 107_429_888),
        punctuation("punctuation-q8", "Q8", 58_527_744, isDefault: true),
        punctuation("punctuation-q6", "Q6", 45_486_080),
        punctuation("punctuation-q5", "Q5", 38_965_248),
        punctuation("punctuation-q4", "Q4", 32_440_320),
    ]

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
}

public enum GraniteModelCache {
    public static var rootDirectory: URL {
        let hub = HubApi()
        return hub.localRepoLocation(.init(id: "placeholder/repository"))
            .deletingLastPathComponent().deletingLastPathComponent()
    }

    public static func directory(for repositoryID: String) throws -> URL {
        guard isValidRepositoryID(repositoryID) else {
            throw GraniteModelManagementError.invalidRepositoryID(repositoryID)
        }
        return HubApi().localRepoLocation(.init(id: repositoryID)).standardizedFileURL
    }

    public static func isDownloaded(_ repositoryID: String, kind: GraniteManagedModelKind? = nil) -> Bool {
        guard let directory = try? directory(for: repositoryID) else { return false }
        return detectedKind(at: directory).map { kind == nil || $0 == kind } ?? false
    }

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
                    catalogAlias: GraniteModelCatalog.model(for: repositoryID)?.alias))
            }
        }
        return result.sorted { $0.repositoryID.localizedCaseInsensitiveCompare($1.repositoryID) == .orderedAscending }
    }

    @discardableResult
    public static func download(
        _ aliasOrID: String,
        kind requestedKind: GraniteManagedModelKind? = nil,
        hfToken: String? = nil,
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
        if isDownloaded(id, kind: kind) {
            event(.cacheHit, 1, nil)
            return destination
        }
        event(.checking, 0, nil)
        try preflightDiskSpace(at: destination, expectedBytes: expectedBytes)
        let hub = HubApi(hfToken: hfToken)
        let patterns = kind == .punctuation
            ? ["*.safetensors", "*.json", "*.model", "*.yaml"]
            : ["*.safetensors", "*.json", "*.txt", "*.model"]
        let downloaded = try GraniteModelLoader.runBlocking {
            try await hub.snapshot(from: id, matching: patterns) { progress, speed in
                event(.downloading, progress.fractionCompleted, speed)
            }
        }
        guard detectedKind(at: downloaded) == kind else {
            throw GraniteRecognizerError.invalidModel(downloaded)
        }
        event(.complete, 1, nil)
        return downloaded
    }

    /// Permanently removes one exact materialized Hub repository directory.
    /// It can be restored by downloading the repository again.
    @discardableResult
    public static func remove(_ aliasOrID: String) throws -> GraniteCachedModel {
        let resolved = try GraniteModelCatalog.resolve(aliasOrID)
        let destination = try directory(for: resolved.id)
        guard let kind = detectedKind(at: destination) else {
            throw GraniteModelManagementError.notDownloaded(resolved.id)
        }
        let record = GraniteCachedModel(
            repositoryID: resolved.id, kind: kind, directory: destination,
            sizeBytes: directorySize(destination), catalogAlias: resolved.model?.alias)
        try FileManager.default.removeItem(at: destination)
        return record
    }

    public static func directorySize(_ directory: URL) -> Int64 {
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .fileSizeKey, .isSymbolicLinkKey]
        guard let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: Array(keys),
            options: [.skipsPackageDescendants]) else { return 0 }
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
        let weights = directory.appendingPathComponent("model.safetensors")
        guard manager.fileExists(atPath: weights.path) else { return nil }
        let tokenizer = directory.appendingPathComponent("tokenizer.json")
        guard manager.fileExists(atPath: tokenizer.path) else { return nil }
        let punctuationConfig = directory.appendingPathComponent("mlx_config.json")
        if let data = try? Data(contentsOf: punctuationConfig),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           object["architecture"] as? String == "bert-punctuation-capitalization-segmentation" {
            return .punctuation
        }
        let speechConfig = directory.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: speechConfig),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           object["model_type"] as? String == "granite_speech5_ctc" {
            return .speech
        }
        return nil
    }

    private static func inferredKind(from repositoryID: String) -> GraniteManagedModelKind {
        repositoryID.localizedCaseInsensitiveContains("punctuation") ? .punctuation : .speech
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
