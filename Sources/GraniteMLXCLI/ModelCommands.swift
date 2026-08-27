import ArgumentParser
import Darwin
import Foundation
import GraniteMLX

final class ConsoleDownloadProgressReporter: @unchecked Sendable {
    private let lock = NSLock()
    private let interactive = isatty(STDERR_FILENO) != 0
    private let showCacheHits: Bool
    private var announced: Set<String> = []
    private var lastMilestone: [String: Int] = [:]
    private var lineIsActive = false

    init(showCacheHits: Bool = false) {
        self.showCacheHits = showCacheHits
    }

    lazy var handler: GraniteModelDownloadProgressHandler = { [weak self] progress in
        self?.update(progress)
    }

    private func update(_ progress: GraniteModelDownloadProgress) {
        lock.lock()
        defer { lock.unlock() }
        switch progress.phase {
        case .checking:
            announceIfNeeded(progress)
            stderr("Resolving repository files…")
        case .cacheHit:
            if showCacheHits {
                finishInteractiveLine()
                stderr("Already downloaded: \(progress.repositoryID)")
                stderr("Cache: \(progress.cacheDirectory.path)")
            }
        case .downloading:
            announceIfNeeded(progress)
            let percent = min(100, max(0, Int((progress.fractionCompleted * 100).rounded())))
            if interactive {
                let width = 30
                let filled = min(width, max(0, Int(progress.fractionCompleted * Double(width))))
                let bar = String(repeating: "=", count: filled)
                    + String(repeating: " ", count: width - filled)
                var line = String(format: "\r[%@] %3d%%", bar, percent)
                if let total = progress.estimatedTotalBytes {
                    line += " of \(formatBytes(total))"
                }
                if let speed = progress.bytesPerSecond, speed > 0 {
                    line += "  \(formatBytes(Int64(speed)))/s"
                }
                writeStderr(line)
                lineIsActive = true
            } else {
                let milestone = percent / 10
                if lastMilestone[progress.repositoryID] != milestone {
                    lastMilestone[progress.repositoryID] = milestone
                    var line = "Download \(progress.repositoryID): \(percent)%"
                    if let speed = progress.bytesPerSecond, speed > 0 {
                        line += " at \(formatBytes(Int64(speed)))/s"
                    }
                    stderr(line)
                }
            }
        case .complete:
            finishInteractiveLine()
            stderr("Downloaded: \(progress.repositoryID)")
            let size = GraniteModelCache.directorySize(progress.cacheDirectory)
            stderr("Stored \(formatBytes(size)) at \(progress.cacheDirectory.path)")
        }
    }

    private func announceIfNeeded(_ progress: GraniteModelDownloadProgress) {
        guard announced.insert(progress.repositoryID).inserted else { return }
        finishInteractiveLine()
        stderr("Downloading \(progress.kind.rawValue) model: \(progress.repositoryID)")
        if let bytes = progress.estimatedTotalBytes {
            stderr("Expected size: approximately \(formatBytes(bytes))")
        }
        stderr("Cache: \(progress.cacheDirectory.path)")
    }

    private func finishInteractiveLine() {
        guard lineIsActive else { return }
        writeStderr("\n")
        lineIsActive = false
    }
}

struct ModelsCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "models",
        abstract: "List, download, and remove Granite-MLX models.",
        subcommands: [ModelsListCommand.self, ModelsDownloadCommand.self, ModelsRemoveCommand.self],
        defaultSubcommand: ModelsListCommand.self)
}

private struct ModelListRecord: Encodable {
    let alias: String?
    let repositoryID: String
    let kind: String
    let family: String?
    let precision: String?
    let isDefault: Bool
    let isDownloaded: Bool
    let cacheState: String
    let stateDetails: String?
    let expectedBytes: Int64?
    let downloadedBytes: Int64?
    let cacheDirectory: String
}

struct ModelsListCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list",
        abstract: "List available models, download status, and disk usage.")

    @Flag(help: "Show only complete or partial model cache entries.")
    var downloadedOnly = false

    @Flag(help: "Write machine-readable JSON.")
    var json = false

    func run() throws {
        let cached = Dictionary(
            uniqueKeysWithValues: GraniteModelCache.downloadedModels().map { ($0.repositoryID.lowercased(), $0) })
        var records = GraniteModelCatalog.models.compactMap { model -> ModelListRecord? in
            let installed = cached[model.repositoryID.lowercased()]
            if downloadedOnly, installed == nil { return nil }
            return ModelListRecord(
                alias: model.alias, repositoryID: model.repositoryID,
                kind: model.kind.rawValue, family: model.family,
                precision: model.precision, isDefault: model.isDefault,
                isDownloaded: installed?.state == .downloaded,
                cacheState: installed?.state.rawValue ?? GraniteModelCacheState.absent.rawValue,
                stateDetails: installed?.stateDetails,
                expectedBytes: model.expectedBytes,
                downloadedBytes: installed?.sizeBytes,
                cacheDirectory: (try? GraniteModelCache.directory(for: model.repositoryID).path) ?? "")
        }
        let catalogIDs = Set(GraniteModelCatalog.models.map { $0.repositoryID.lowercased() })
        records.append(contentsOf: cached.values.compactMap { installed in
            guard !catalogIDs.contains(installed.repositoryID.lowercased()) else { return nil }
            return ModelListRecord(
                alias: installed.catalogAlias, repositoryID: installed.repositoryID,
                kind: installed.kind.rawValue, family: nil, precision: nil,
                isDefault: false, isDownloaded: installed.state == .downloaded,
                cacheState: installed.state.rawValue,
                stateDetails: installed.stateDetails,
                expectedBytes: nil,
                downloadedBytes: installed.sizeBytes,
                cacheDirectory: installed.directory.path)
        })

        if json {
            let encoder = JSONEncoder()
            encoder.keyEncodingStrategy = .convertToSnakeCase
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            print(String(decoding: try encoder.encode(records), as: UTF8.self))
            return
        }

        print("Granite-MLX model cache: \(GraniteModelCache.rootDirectory.path)")
        print("")
        print("\(padded("STATE", to: 7)) \(padded("DEF", to: 4)) \(padded("TYPE", to: 11)) \(padded("PRECISION", to: 9)) \(padded("SIZE", to: 10)) MODEL")
        for record in records {
            let status = switch record.cacheState {
            case GraniteModelCacheState.downloaded.rawValue: "[x]"
            case GraniteModelCacheState.partial.rawValue: "[-]"
            default: "[ ]"
            }
            let defaultMarker = record.isDefault ? "yes" : ""
            let bytes = record.downloadedBytes ?? record.expectedBytes ?? 0
            let size = bytes > 0 ? formatBytes(bytes) : "unknown"
            let name = record.alias.map { "\($0)  [\(record.repositoryID)]" } ?? record.repositoryID
            print("\(padded(status, to: 7)) \(padded(defaultMarker, to: 4)) \(padded(record.kind, to: 11)) \(padded(record.precision ?? "custom", to: 9)) \(padded(size, to: 10)) \(name)")
            if record.cacheState == GraniteModelCacheState.partial.rawValue,
               let details = record.stateDetails {
                print("        Repair with `granite-mlx models download \(record.alias ?? record.repositoryID)`. Details: \(details)")
            }
        }
        let total = cached.values.reduce(Int64(0)) { $0 + $1.sizeBytes }
        print("")
        let completeCount = cached.values.filter { $0.state == .downloaded }.count
        let partialCount = cached.values.filter { $0.state == .partial }.count
        print("Cache: \(completeCount) complete, \(partialCount) partial, \(formatBytes(total)) total")
        print("Use `granite-mlx models download <alias>` or `granite-mlx models remove <alias>`." )
    }
}

struct ModelsDownloadCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "download",
        abstract: "Download one or more models into the local cache.")

    @Argument(help: "Catalog alias or Hugging Face repository ID.")
    var models: [String]

    @Option(help: "Hugging Face access token for private or gated repositories; overrides HF_TOKEN.")
    var hfToken: String?

    func validate() throws {
        guard !models.isEmpty else {
            throw ValidationError("[GMLX-CLI-101] Provide at least one model alias or repository ID.")
        }
    }

    func run() throws {
        let reporter = ConsoleDownloadProgressReporter(showCacheHits: true)
        let effectiveHFToken = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
        for model in models {
            do {
                _ = try GraniteModelCache.download(
                    model, hfToken: effectiveHFToken,
                    progressHandler: reporter.handler)
            } catch {
                throw CLIRuntimeError.wrapping(
                    error, code: "GMLX-CLI-102", operation: "Downloading model \(model)")
            }
        }
    }
}

struct ModelsRemoveCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "remove",
        abstract: "Remove downloaded models to reclaim disk space.")

    @Argument(help: "Catalog alias or Hugging Face repository ID.")
    var models: [String] = []

    @Flag(help: "Remove every downloaded Granite-compatible model.")
    var all = false

    @Flag(name: [.short, .long], help: "Remove without asking for confirmation.")
    var yes = false

    func validate() throws {
        guard all != !models.isEmpty else {
            throw ValidationError("[GMLX-CLI-103] Specify model names or --all, but not both.")
        }
    }

    func run() throws {
        let targets: [GraniteCachedModel]
        if all {
            targets = GraniteModelCache.downloadedModels()
        } else {
            let downloaded = GraniteModelCache.downloadedModels()
            targets = try models.map { value in
                let id = try GraniteModelCatalog.resolve(value).id
                guard let record = downloaded.first(where: {
                    $0.repositoryID.caseInsensitiveCompare(id) == .orderedSame
                }) else { throw GraniteModelManagementError.notDownloaded(id) }
                return record
            }
        }
        guard !targets.isEmpty else {
            print("No downloaded Granite-MLX models to remove.")
            return
        }
        let bytes = targets.reduce(Int64(0)) { $0 + $1.sizeBytes }
        if !yes {
            guard isatty(STDIN_FILENO) != 0 else {
                throw ValidationError("[GMLX-CLI-104] Removal requires --yes when stdin is not interactive.")
            }
            print("Remove \(targets.count) model(s) and reclaim \(formatBytes(bytes))? [y/N] ", terminator: "")
            guard let response = readLine()?.lowercased(), response == "y" || response == "yes" else {
                print("Cancelled.")
                return
            }
        }
        for target in targets {
            let removed = try GraniteModelCache.remove(target.repositoryID)
            print("Removed \(removed.repositoryID) (\(formatBytes(removed.sizeBytes))).")
        }
        print("Reclaimed \(formatBytes(bytes)). Models can be downloaded again at any time.")
    }
}

func formatBytes(_ bytes: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
}

private func writeStderr(_ text: String) {
    FileHandle.standardError.write(Data(text.utf8))
}

private func padded(_ value: String, to width: Int) -> String {
    value + String(repeating: " ", count: max(0, width - value.count))
}
