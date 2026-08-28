import Foundation

/// Explicit storage locations used by GraniteMLX model acquisition and Core ML compilation.
///
/// Pass one value through model browsing, downloading, removal, and recognizer
/// construction so every operation addresses the same files. This type uses only
/// filesystem URLs and is suitable for macOS, iOS, and iPadOS apps.
public struct GraniteModelStorage: Sendable, Equatable {
    /// Parent directory in which Hugging Face materializes its `models` directory.
    ///
    /// A repository named `owner/repository` is stored at
    /// `hubDirectory/models/owner/repository`.
    public let hubDirectory: URL

    /// Content-addressed Hugging Face transfer cache, or `nil` to disable it.
    ///
    /// The cache can avoid repeated network transfers and can be shared with other
    /// Hugging Face clients when they use the same location.
    public let downloadCacheDirectory: URL?

    /// Persistent location for device-specific compiled Core ML models, or `nil`
    /// to compile into temporary storage for each recognizer lifetime.
    public let compiledCoreMLDirectory: URL?

    /// Directory containing materialized `owner/repository` model folders.
    public var modelsDirectory: URL {
        hubDirectory.appendingPathComponent("models", isDirectory: true)
    }

    /// Creates an explicit model-storage configuration.
    /// - Parameters:
    ///   - hubDirectory: Parent directory in which a `models` directory is created.
    ///   - downloadCacheDirectory: Shared transfer cache, or `nil` to disable it.
    ///   - compiledCoreMLDirectory: Persistent compiled Core ML cache, or `nil`
    ///     to use temporary compilation output.
    public init(
        hubDirectory: URL,
        downloadCacheDirectory: URL?,
        compiledCoreMLDirectory: URL?
    ) {
        self.hubDirectory = hubDirectory.standardizedFileURL
        self.downloadCacheDirectory = downloadCacheDirectory?.standardizedFileURL
        self.compiledCoreMLDirectory = compiledCoreMLDirectory?.standardizedFileURL
    }

    /// Platform-appropriate locations without considering environment variables.
    ///
    /// On sandboxed Apple platforms these URLs are inside the app container.
    public static var platformDefault: GraniteModelStorage {
        let manager = FileManager.default
        let documents = manager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let caches = manager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return GraniteModelStorage(
            hubDirectory: documents.appendingPathComponent("huggingface", isDirectory: true),
            downloadCacheDirectory: defaultDownloadCacheDirectory,
            compiledCoreMLDirectory: caches.appendingPathComponent(
                "GraniteMLX/CoreML", isDirectory: true))
    }

    /// Default locations with optional command-line environment overrides applied.
    ///
    /// Applications should normally construct and retain an explicit value. The
    /// environment behavior exists for command-line compatibility and is not
    /// required to access any library capability.
    public static var `default`: GraniteModelStorage {
        let environment = ProcessInfo.processInfo.environment
        let base = platformDefault
        let hub = environment["GRANITE_MLX_HUB_DIRECTORY"].map {
            directoryURL(for: $0)
        } ?? base.hubDirectory
        let transfer = environment["HF_HUB_CACHE"].map {
            directoryURL(for: $0)
        } ?? environment["HF_HOME"].map {
            directoryURL(for: $0)
                .appendingPathComponent("hub", isDirectory: true)
        } ?? base.downloadCacheDirectory
        let compiled = environment["GRANITE_MLX_COREML_CACHE_DIRECTORY"].map {
            directoryURL(for: $0)
        } ?? base.compiledCoreMLDirectory
        return GraniteModelStorage(
            hubDirectory: hub,
            downloadCacheDirectory: transfer,
            compiledCoreMLDirectory: compiled)
    }

    private static var defaultDownloadCacheDirectory: URL {
        #if os(macOS)
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
        #else
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("huggingface/hub", isDirectory: true)
        #endif
    }

    private static func directoryURL(for path: String) -> URL {
        URL(
            fileURLWithPath: NSString(string: path).expandingTildeInPath,
            isDirectory: true)
    }
}

/// Application-facing model manager bound to one explicit storage configuration.
///
/// Use this value to present the published catalog, inspect local state, download
/// checkpoints, and remove them. Every operation is guaranteed to address the
/// same configured directories.
public struct GraniteModelManager: Sendable {
    /// Storage locations used by every operation on this manager.
    public let storage: GraniteModelStorage

    /// Models published and supported by this GraniteMLX release.
    public var availableModels: [GranitePublishedModel] {
        GraniteModelCatalog.models
    }

    /// Root containing materialized model repositories.
    public var modelsDirectory: URL {
        GraniteModelCache.rootDirectory(storage: storage)
    }

    /// Creates a model manager bound to explicit storage.
    /// - Parameter storage: Application-owned model and cache locations.
    public init(storage: GraniteModelStorage) {
        self.storage = storage
    }

    /// Creates a manager using the command-line-compatible default locations.
    public init() {
        self.init(storage: .default)
    }

    /// Returns the materialized directory for a repository without creating it.
    /// - Parameter repositoryID: Full `owner/repository` identifier.
    /// - Returns: Repository directory beneath ``modelsDirectory``.
    /// - Throws: ``GraniteModelManagementError/invalidRepositoryID(_:)`` for an
    ///   unsafe or malformed identifier.
    public func directory(for repositoryID: String) throws -> URL {
        try GraniteModelCache.directory(for: repositoryID, storage: storage)
    }

    /// Returns the validated local state of a repository.
    /// - Parameters:
    ///   - repositoryID: Full `owner/repository` identifier.
    ///   - kind: Expected model role, or `nil` to accept any supported role.
    /// - Returns: Absent, partial, or completely downloaded state.
    public func state(
        of repositoryID: String,
        kind: GraniteManagedModelKind? = nil
    ) -> GraniteModelCacheState {
        GraniteModelCache.state(
            of: repositoryID, kind: kind, storage: storage)
    }

    /// Indicates whether a complete compatible checkpoint is available locally.
    /// - Parameters:
    ///   - repositoryID: Full `owner/repository` identifier.
    ///   - kind: Expected model role, or `nil` for any supported role.
    /// - Returns: `true` only for a complete, validated checkpoint.
    public func isDownloaded(
        _ repositoryID: String,
        kind: GraniteManagedModelKind? = nil
    ) -> Bool {
        GraniteModelCache.isDownloaded(
            repositoryID, kind: kind, storage: storage)
    }

    /// Lists complete and partial compatible checkpoints in this manager's storage.
    /// - Returns: Cached-model records sorted by repository identifier.
    public func downloadedModels() -> [GraniteCachedModel] {
        GraniteModelCache.downloadedModels(storage: storage)
    }

    /// Returns shared Hugging Face transfer-cache usage for one repository.
    /// - Parameter repositoryID: Full `owner/repository` identifier.
    /// - Returns: Logical bytes retained for transfer reuse.
    public func downloadCacheSize(for repositoryID: String) -> Int64 {
        GraniteModelCache.downloadCacheSize(
            for: repositoryID, storage: storage)
    }

    /// Removes shared transfer-cache data for one repository without deleting
    /// its materialized, inference-ready model directory.
    /// - Parameter repositoryID: Full `owner/repository` identifier.
    /// - Returns: Logical bytes present before removal.
    /// - Throws: Invalid-identifier or filesystem errors.
    @discardableResult
    public func removeDownloadCache(for repositoryID: String) throws -> Int64 {
        try GraniteModelCache.removeDownloadCache(
            for: repositoryID, storage: storage)
    }

    /// Downloads or resumes and validates one checkpoint.
    /// - Parameters:
    ///   - aliasOrID: Published alias or full Hugging Face repository identifier.
    ///   - kind: Explicit role for a custom repository, or `nil` to infer it.
    ///   - hfToken: Optional token for private or gated repositories.
    ///   - cancellationToken: Cooperative cancellation token.
    ///   - progressHandler: Receives cache-check and transfer progress.
    /// - Returns: Materialized, validated repository directory.
    /// - Throws: Model-management or cancellation errors with stable diagnostics.
    @discardableResult
    public func download(
        _ aliasOrID: String,
        kind: GraniteManagedModelKind? = nil,
        hfToken: String? = nil,
        cancellationToken: GraniteCancellationToken? = nil,
        progressHandler: GraniteModelDownloadProgressHandler? = nil
    ) throws -> URL {
        try GraniteModelCache.download(
            aliasOrID, kind: kind, storage: storage, hfToken: hfToken,
            cancellationToken: cancellationToken,
            progressHandler: progressHandler)
    }

    /// Permanently removes one materialized checkpoint plus its transfer and
    /// compiled Core ML caches.
    /// - Parameter aliasOrID: Published alias or exact repository identifier.
    /// - Returns: Metadata describing the removed files and reclaimed size.
    /// - Throws: Model-management errors when the model is absent or removal fails.
    @discardableResult
    public func remove(_ aliasOrID: String) throws -> GraniteCachedModel {
        try GraniteModelCache.remove(aliasOrID, storage: storage)
    }
}
