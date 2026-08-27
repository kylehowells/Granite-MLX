import ArgumentParser
import Foundation
import GraniteMLX

struct GraniteCLIConfiguration: Codable, Equatable {
    static let currentSchemaVersion = 1

    let schemaVersion: Int
    var defaultBackend: GraniteSpeechBackend?

    init(defaultBackend: GraniteSpeechBackend? = nil) {
        self.schemaVersion = Self.currentSchemaVersion
        self.defaultBackend = defaultBackend
    }
}

enum CLIConfigurationError: Error, LocalizedError {
    case unreadable(URL, details: String)
    case invalid(URL, details: String)
    case unwritable(URL, details: String)
    case removalFailed(URL, details: String)

    var code: String {
        switch self {
        case .unreadable: "GMLX-CONFIG-001"
        case .invalid: "GMLX-CONFIG-002"
        case .unwritable: "GMLX-CONFIG-003"
        case .removalFailed: "GMLX-CONFIG-004"
        }
    }

    var errorDescription: String? {
        switch self {
        case .unreadable(let url, let details):
            "[\(code)] Could not read the Granite-MLX configuration. Technical details: path=\(url.path); underlying=\(details)"
        case .invalid(let url, let details):
            "[\(code)] The Granite-MLX configuration is invalid. Repair it with `granite-mlx config set backend mlx`, or remove the saved setting with `granite-mlx config unset backend`. Technical details: path=\(url.path); underlying=\(details)"
        case .unwritable(let url, let details):
            "[\(code)] Could not save the Granite-MLX configuration. Check directory permissions and available disk space. Technical details: path=\(url.path); underlying=\(details)"
        case .removalFailed(let url, let details):
            "[\(code)] Could not remove the saved Granite-MLX backend setting. Technical details: path=\(url.path); underlying=\(details)"
        }
    }
}

enum CLIConfigurationStore {
    static var fileURL: URL {
        if let testPath = ProcessInfo.processInfo.environment[
            "GRANITE_MLX_TEST_CONFIG_PATH"] {
            return URL(fileURLWithPath: testPath).standardizedFileURL
        }
        return FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Granite-MLX", isDirectory: true)
            .appendingPathComponent("config.json")
    }

    static func load() throws -> GraniteCLIConfiguration {
        let url = fileURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            return GraniteCLIConfiguration()
        }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw CLIConfigurationError.unreadable(
                url, details: String(reflecting: error))
        }
        do {
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            let configuration = try decoder.decode(
                GraniteCLIConfiguration.self, from: data)
            guard configuration.schemaVersion == GraniteCLIConfiguration.currentSchemaVersion else {
                throw CLIConfigurationError.invalid(
                    url,
                    details: "schema_version=\(configuration.schemaVersion); supported_schema_version=\(GraniteCLIConfiguration.currentSchemaVersion)")
            }
            return configuration
        } catch let error as CLIConfigurationError {
            throw error
        } catch {
            throw CLIConfigurationError.invalid(
                url, details: String(reflecting: error))
        }
    }

    static func save(defaultBackend: GraniteSpeechBackend) throws {
        let url = fileURL
        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.keyEncodingStrategy = .convertToSnakeCase
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            var data = try encoder.encode(
                GraniteCLIConfiguration(defaultBackend: defaultBackend))
            data.append(0x0A)
            try data.write(to: url, options: .atomic)
        } catch {
            throw CLIConfigurationError.unwritable(
                url, details: String(reflecting: error))
        }
    }

    static func unsetBackend() throws -> Bool {
        let url = fileURL
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        do {
            try FileManager.default.removeItem(at: url)
            return true
        } catch {
            throw CLIConfigurationError.removalFailed(
                url, details: String(reflecting: error))
        }
    }
}

struct ConfigCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "config",
        abstract: "View or change persistent Granite-MLX user settings.",
        discussion: """
        EXAMPLES:
          granite-mlx config show
          granite-mlx config get backend
          granite-mlx config set backend coreml
          granite-mlx config set backend mlx
          granite-mlx config unset backend

        Settings are stored in ~/Library/Application Support/Granite-MLX/config.json.
        An explicit transcription option such as --backend always overrides the
        saved setting for that invocation.
        """,
        subcommands: [
            ConfigShowCommand.self,
            ConfigGetCommand.self,
            ConfigSetCommand.self,
            ConfigUnsetCommand.self,
        ],
        defaultSubcommand: ConfigShowCommand.self)
}

private struct ConfigurationRecord: Encodable {
    let configFile: String
    let backend: String
    let source: String
}

struct ConfigShowCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "show",
        abstract: "Show saved settings and their configuration file.")

    @Flag(help: "Write machine-readable JSON.")
    var json = false

    func run() throws {
        let saved = try CLIConfigurationStore.load()
        let record = ConfigurationRecord(
            configFile: CLIConfigurationStore.fileURL.path,
            backend: saved.defaultBackend?.rawValue ?? GraniteSpeechBackend.mlx.rawValue,
            source: saved.defaultBackend == nil ? "built-in" : "saved")
        if json {
            let encoder = JSONEncoder()
            encoder.keyEncodingStrategy = .convertToSnakeCase
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            print(String(decoding: try encoder.encode(record), as: UTF8.self))
        } else {
            print("Configuration: \(record.configFile)")
            print("Default backend: \(record.backend) (\(record.source))")
        }
    }
}

struct ConfigGetCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "get",
        abstract: "Print one effective setting value.")

    @Argument(help: "Setting name; currently backend.")
    var key: String

    func validate() throws {
        try validateBackendKey(key)
    }

    func run() throws {
        let saved = try CLIConfigurationStore.load()
        print((saved.defaultBackend ?? .mlx).rawValue)
    }
}

struct ConfigSetCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "set",
        abstract: "Persist a user setting.",
        discussion: """
        EXAMPLES:
          granite-mlx config set backend coreml
          granite-mlx config set backend mlx
        """)

    @Argument(help: "Setting name; currently backend.")
    var key: String

    @Argument(help: "Setting value; backend accepts mlx or coreml.")
    var value: String

    func validate() throws {
        try validateBackendKey(key)
        guard GraniteSpeechBackend(rawValue: value.lowercased()) != nil else {
            throw ValidationError(
                "[GMLX-CONFIG-006] Unsupported backend `\(value)`. Use mlx or coreml.")
        }
    }

    func run() throws {
        let backend = GraniteSpeechBackend(rawValue: value.lowercased())!
        try CLIConfigurationStore.save(defaultBackend: backend)
        print("Default backend set to \(backend.rawValue).")
        print("Saved in \(CLIConfigurationStore.fileURL.path)")
    }
}

struct ConfigUnsetCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "unset",
        abstract: "Remove a saved setting and restore its built-in default.")

    @Argument(help: "Setting name; currently backend.")
    var key: String

    func validate() throws {
        try validateBackendKey(key)
    }

    func run() throws {
        if try CLIConfigurationStore.unsetBackend() {
            print("Removed the saved backend setting. Default backend is now mlx.")
        } else {
            print("No backend setting was saved. Default backend is mlx.")
        }
    }
}

private func validateBackendKey(_ key: String) throws {
    guard ["backend", "default-backend"].contains(key.lowercased()) else {
        throw ValidationError(
            "[GMLX-CONFIG-005] Unknown setting `\(key)`. The supported setting is backend.")
    }
}
