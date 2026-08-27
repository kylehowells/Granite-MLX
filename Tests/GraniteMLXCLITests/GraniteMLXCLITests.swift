import Darwin
import Foundation
import XCTest

final class GraniteMLXCLITests: XCTestCase {
    private struct Result {
        let status: Int32
        let stdout: String
        let stderr: String
    }

    private final class PipeCollector: @unchecked Sendable {
        private let handle: FileHandle
        private let group = DispatchGroup()
        private var data = Data()

        init(_ handle: FileHandle) {
            self.handle = handle
        }

        func start() {
            group.enter()
            DispatchQueue.global(qos: .utility).async { [self] in
                data = handle.readDataToEndOfFile()
                group.leave()
            }
        }

        func text() -> String {
            group.wait()
            return String(decoding: data, as: UTF8.self)
        }
    }

    private var executable: URL {
        Bundle(for: Self.self).bundleURL
            .deletingLastPathComponent()
            .appendingPathComponent("granite-mlx")
    }

    private func runCLI(
        _ arguments: [String],
        environment additions: [String: String] = [:],
        timeout: TimeInterval = 60,
        interruptAfter: TimeInterval? = nil
    ) throws -> Result {
        let process = Process()
        process.executableURL = executable
        process.arguments = arguments
        var environment = ProcessInfo.processInfo.environment.merging(additions) { _, new in new }
        if additions["GRANITE_MLX_TEST_CONFIG_PATH"] == nil {
            environment["GRANITE_MLX_TEST_CONFIG_PATH"] = FileManager.default
                .temporaryDirectory
                .appendingPathComponent(
                    "granite-mlx-tests-\(ProcessInfo.processInfo.processIdentifier)-absent-config.json")
                .path
        }
        process.environment = environment
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        let stdout = PipeCollector(stdoutPipe.fileHandleForReading)
        let stderr = PipeCollector(stderrPipe.fileHandleForReading)
        stdout.start()
        stderr.start()
        try process.run()
        if let interruptAfter {
            DispatchQueue.global().asyncAfter(deadline: .now() + interruptAfter) {
                if process.isRunning { kill(process.processIdentifier, SIGINT) }
            }
        }
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning, Date() < deadline {
            Thread.sleep(forTimeInterval: 0.02)
        }
        if process.isRunning {
            process.terminate()
            process.waitUntilExit()
            XCTFail("CLI timed out after \(timeout)s: \(arguments.joined(separator: " "))")
        } else {
            process.waitUntilExit()
        }
        return Result(status: process.terminationStatus, stdout: stdout.text(), stderr: stderr.text())
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("granite-mlx-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func writeSilentWAV(to url: URL, seconds: Int = 2) throws {
        let sampleRate: UInt32 = 16_000
        let sampleCount = sampleRate * UInt32(seconds)
        let dataBytes = sampleCount * 2
        var data = Data()
        func append<T>(_ value: T) {
            var littleEndian = value
            withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
        }
        data.append(contentsOf: "RIFF".utf8)
        append(UInt32(36) + dataBytes)
        data.append(contentsOf: "WAVEfmt ".utf8)
        append(UInt32(16))
        append(UInt16(1))
        append(UInt16(1))
        append(sampleRate)
        append(sampleRate * 2)
        append(UInt16(2))
        append(UInt16(16))
        data.append(contentsOf: "data".utf8)
        append(dataBytes)
        data.append(Data(count: Int(dataBytes)))
        try data.write(to: url)
    }

    private func realModelArguments() throws -> [String] {
        guard let model = ProcessInfo.processInfo.environment["GRANITE_TEST_SPEECH_MODEL"] else {
            throw XCTSkip("Set GRANITE_TEST_SPEECH_MODEL to run real-model CLI integration tests.")
        }
        var arguments = ["--model", model]
        if let punctuation = ProcessInfo.processInfo.environment["GRANITE_TEST_PUNCTUATION_MODEL"] {
            arguments += ["--punctuation-model", punctuation]
        } else {
            arguments.append("--no-punctuate")
        }
        return arguments
    }

    func testRootHelpVersionAndValidationAreOffline() throws {
        let help = try runCLI(["--help"])
        XCTAssertEqual(help.status, 0)
        XCTAssertTrue(help.stdout.contains("COMMON EXAMPLES"))
        XCTAssertTrue(help.stdout.contains("models download"))
        XCTAssertTrue(help.stdout.contains("config set backend coreml"))
        XCTAssertFalse(help.stdout.contains("GRANITE_MLX_BACKEND"))
        XCTAssertEqual(help.stderr, "")

        let version = try runCLI(["--version"])
        XCTAssertEqual(version.status, 0)
        XCTAssertEqual(version.stdout.trimmingCharacters(in: .whitespacesAndNewlines), "0.1.0")
        XCTAssertEqual(version.stderr, "")

        let invalid = try runCLI(["file.wav", "--output-format", "xml"])
        XCTAssertNotEqual(invalid.status, 0)
        XCTAssertEqual(invalid.stdout, "")
        XCTAssertTrue(invalid.stderr.contains("GMLX-CLI-002"))

        let conflictingCoreMLModel = try runCLI([
            "file.wav", "--backend", "mlx", "--coreml-model", "Granite.mlpackage",
        ])
        XCTAssertNotEqual(conflictingCoreMLModel.status, 0)
        XCTAssertTrue(conflictingCoreMLModel.stderr.contains("GMLX-CLI-014"))

        let invalidBackend = try runCLI([
            "file.wav", "--backend", "tensor-core",
        ])
        XCTAssertNotEqual(invalidBackend.status, 0)
        XCTAssertTrue(invalidBackend.stderr.contains("GMLX-CLI-013"))

        let legacyEnvironmentIsIgnored = try runCLI([
            "file.wav", "--model", "apache-coreml-q8",
        ], environment: ["GRANITE_MLX_BACKEND": "coreml"])
        XCTAssertNotEqual(legacyEnvironmentIsIgnored.status, 0)
        XCTAssertTrue(legacyEnvironmentIsIgnored.stderr.contains("GMLX-CLI-017"))
        XCTAssertTrue(legacyEnvironmentIsIgnored.stderr.contains("backend `mlx`"))

        let invalidComputeUnits = try runCLI([
            "file.wav", "--backend", "coreml", "--coreml-model", "Granite.mlpackage",
            "--coreml-compute-units", "gpu-only",
        ])
        XCTAssertNotEqual(invalidComputeUnits.status, 0)
        XCTAssertTrue(invalidComputeUnits.stderr.contains("GMLX-CLI-015"))

        let mismatchedBackendModel = try runCLI([
            "file.wav", "--backend", "coreml", "--model", "apache-q8",
        ])
        XCTAssertNotEqual(mismatchedBackendModel.status, 0)
        XCTAssertTrue(mismatchedBackendModel.stderr.contains("GMLX-CLI-017"))
        XCTAssertTrue(mismatchedBackendModel.stderr.contains("--backend mlx"))
    }

    func testPersistentBackendConfigurationLifecycleAndPrecedence() throws {
        let temporary = try temporaryDirectory()
        let config = temporary.appendingPathComponent("settings/config.json")
        let environment = ["GRANITE_MLX_TEST_CONFIG_PATH": config.path]

        let initial = try runCLI(["config", "show"], environment: environment)
        XCTAssertEqual(initial.status, 0, initial.stderr)
        XCTAssertTrue(initial.stdout.contains("Default backend: mlx (built-in)"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: config.path))

        let saved = try runCLI(
            ["config", "set", "backend", "coreml"], environment: environment)
        XCTAssertEqual(saved.status, 0, saved.stderr)
        XCTAssertTrue(saved.stdout.contains("Default backend set to coreml"))
        let stored = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: config)) as? [String: Any])
        XCTAssertEqual(stored["schema_version"] as? Int, 1)
        XCTAssertEqual(stored["default_backend"] as? String, "coreml")

        let shown = try runCLI(
            ["config", "show", "--json"], environment: environment)
        XCTAssertEqual(shown.status, 0, shown.stderr)
        let shownRecord = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(shown.stdout.utf8)) as? [String: Any])
        XCTAssertEqual(shownRecord["backend"] as? String, "coreml")
        XCTAssertEqual(shownRecord["source"] as? String, "saved")

        let value = try runCLI(["config", "get", "backend"], environment: environment)
        XCTAssertEqual(value.status, 0, value.stderr)
        XCTAssertEqual(value.stdout.trimmingCharacters(in: .whitespacesAndNewlines), "coreml")

        let savedDefaultApplies = try runCLI([
            "file.wav", "--model", "apache-q8",
        ], environment: environment)
        XCTAssertNotEqual(savedDefaultApplies.status, 0)
        XCTAssertTrue(savedDefaultApplies.stderr.contains("GMLX-CLI-017"))
        XCTAssertTrue(savedDefaultApplies.stderr.contains("backend `coreml`"))

        let explicitBackendWins = try runCLI([
            "file.wav", "--backend", "mlx", "--model", "apache-coreml-q8",
        ], environment: environment)
        XCTAssertNotEqual(explicitBackendWins.status, 0)
        XCTAssertTrue(explicitBackendWins.stderr.contains("GMLX-CLI-017"))
        XCTAssertTrue(explicitBackendWins.stderr.contains("backend `mlx`"))

        let invalidKey = try runCLI(
            ["config", "set", "theme", "coreml"], environment: environment)
        XCTAssertNotEqual(invalidKey.status, 0)
        XCTAssertTrue(invalidKey.stderr.contains("GMLX-CONFIG-005"))
        let invalidValue = try runCLI(
            ["config", "set", "backend", "metal"], environment: environment)
        XCTAssertNotEqual(invalidValue.status, 0)
        XCTAssertTrue(invalidValue.stderr.contains("GMLX-CONFIG-006"))

        let unset = try runCLI(
            ["config", "unset", "backend"], environment: environment)
        XCTAssertEqual(unset.status, 0, unset.stderr)
        XCTAssertFalse(FileManager.default.fileExists(atPath: config.path))
        let restored = try runCLI(
            ["config", "get", "backend"], environment: environment)
        XCTAssertEqual(restored.stdout.trimmingCharacters(in: .whitespacesAndNewlines), "mlx")

        try FileManager.default.createDirectory(
            at: config.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("{broken".utf8).write(to: config)
        let corrupt = try runCLI(["config", "show"], environment: environment)
        XCTAssertNotEqual(corrupt.status, 0)
        XCTAssertTrue(corrupt.stderr.contains("GMLX-CONFIG-002"), corrupt.stderr)
        XCTAssertTrue(corrupt.stderr.contains(config.path), corrupt.stderr)

        let explicitBackendBypassesCorruptSavedConfiguration = try runCLI([
            "file.wav", "--backend", "mlx", "--model", "apache-coreml-q8",
        ], environment: environment)
        XCTAssertNotEqual(explicitBackendBypassesCorruptSavedConfiguration.status, 0)
        XCTAssertTrue(
            explicitBackendBypassesCorruptSavedConfiguration.stderr.contains("GMLX-CLI-017"),
            explicitBackendBypassesCorruptSavedConfiguration.stderr)
        XCTAssertFalse(
            explicitBackendBypassesCorruptSavedConfiguration.stderr.contains("GMLX-CONFIG-002"),
            explicitBackendBypassesCorruptSavedConfiguration.stderr)

        let repaired = try runCLI(
            ["config", "set", "backend", "mlx"], environment: environment)
        XCTAssertEqual(repaired.status, 0, repaired.stderr)
        XCTAssertEqual(
            try runCLI(["config", "get", "backend"], environment: environment)
                .stdout.trimmingCharacters(in: .whitespacesAndNewlines),
            "mlx")
    }

    func testModelListReportsAbsentPartialCompleteAndWarmCache() throws {
        let temporary = try temporaryDirectory()
        let hub = temporary.appendingPathComponent("hub")
        let models = hub.appendingPathComponent("models/iky1e")
        let complete = models.appendingPathComponent("granite-speech-5.0-470m-turboctc-mlx-q8")
        let partial = models.appendingPathComponent("punctuation-fullstop-truecase-english-mlx-q8")
        try FileManager.default.createDirectory(at: complete, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: partial, withIntermediateDirectories: true)
        try writeSafetensorsHeader(
            names: ["encoder.input_linear.weight", "encoder.out.weight"],
            to: complete.appendingPathComponent("model.safetensors"))
        try Data("{\"model_type\":\"granite_speech5_ctc\"}".utf8)
            .write(to: complete.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(to: complete.appendingPathComponent("tokenizer.json"))
        try Data("interrupted".utf8).write(to: partial.appendingPathComponent("model.safetensors"))

        let environment = ["GRANITE_MLX_HUB_DIRECTORY": hub.path]
        let listed = try runCLI(["models", "list", "--json"], environment: environment)
        XCTAssertEqual(listed.status, 0)
        XCTAssertEqual(listed.stderr, "")
        let records = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(listed.stdout.utf8)) as? [[String: Any]])
        let states = Dictionary(uniqueKeysWithValues: records.compactMap { record -> (String, String)? in
            guard let alias = record["alias"] as? String,
                  let state = record["cache_state"] as? String else { return nil }
            return (alias, state)
        })
        XCTAssertEqual(states["apache-q8"], "downloaded")
        XCTAssertEqual(states["punctuation-q8"], "partial")
        XCTAssertEqual(states["apache-q6"], "absent")
        XCTAssertEqual(states["apache-coreml-q8"], "absent")

        let warm = try runCLI(
            ["models", "download", "apache-q8"], environment: environment)
        XCTAssertEqual(warm.status, 0)
        XCTAssertEqual(warm.stdout, "")
        XCTAssertTrue(warm.stderr.contains("Already downloaded"))

        let removed = try runCLI(
            ["models", "remove", "punctuation-q8", "--yes"], environment: environment)
        XCTAssertEqual(removed.status, 0)
        XCTAssertTrue(removed.stdout.contains("Removed"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: partial.path))
    }

    func testAllExportersBatchTemplatesAndCollisions() throws {
        let temporary = try temporaryDirectory()
        let firstDirectory = temporary.appendingPathComponent("first")
        let secondDirectory = temporary.appendingPathComponent("second")
        let output = temporary.appendingPathComponent("output")
        try FileManager.default.createDirectory(at: firstDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: secondDirectory, withIntermediateDirectories: true)
        let first = firstDirectory.appendingPathComponent("sample.wav")
        let second = secondDirectory.appendingPathComponent("sample.wav")
        if let fixture = ProcessInfo.processInfo.environment["GRANITE_TEST_AUDIO"] {
            try FileManager.default.copyItem(at: URL(fileURLWithPath: fixture), to: first)
            try FileManager.default.copyItem(at: URL(fileURLWithPath: fixture), to: second)
        } else {
            try writeSilentWAV(to: first)
            try writeSilentWAV(to: second)
        }
        var arguments = [
            first.path, second.path, "--output-format", "all",
            "--output-dir", output.path, "--output-template", "{filename}",
        ]
        arguments += try realModelArguments()
        let result = try runCLI(arguments, timeout: 180)
        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertEqual(result.stdout, "")
        for ext in ["txt", "srt", "vtt", "json"] {
            XCTAssertTrue(FileManager.default.fileExists(atPath: output.appendingPathComponent("sample.\(ext)").path))
            XCTAssertTrue(FileManager.default.fileExists(atPath: output.appendingPathComponent("sample-2.\(ext)").path))
        }
        let vtt = try String(contentsOf: output.appendingPathComponent("sample.vtt"), encoding: .utf8)
        XCTAssertTrue(vtt.hasPrefix("WEBVTT"))
        let json = try Data(contentsOf: output.appendingPathComponent("sample.json"))
        let document = try XCTUnwrap(JSONSerialization.jsonObject(with: json) as? [String: Any])
        XCTAssertNotNil(document["text"])
        XCTAssertNotNil(document["raw_text"])
        XCTAssertNotNil(document["performance"])
    }

    func testIndividualOutputFormatSelection() throws {
        let temporary = try temporaryDirectory()
        let audio = temporary.appendingPathComponent("sample.wav")
        if let fixture = ProcessInfo.processInfo.environment["GRANITE_TEST_AUDIO"] {
            try FileManager.default.copyItem(at: URL(fileURLWithPath: fixture), to: audio)
        } else {
            try writeSilentWAV(to: audio)
        }
        for format in ["txt", "srt", "vtt", "json"] {
            let output = temporary.appendingPathComponent(format)
            var arguments = [
                audio.path, "--output-format", format,
                "--output-dir", output.path,
            ]
            arguments += try realModelArguments()
            let result = try runCLI(arguments, timeout: 180)
            XCTAssertEqual(result.status, 0, "\(format): \(result.stderr)")
            XCTAssertEqual(result.stdout, "")
            let destination = output.appendingPathComponent("sample.\(format)")
            XCTAssertTrue(FileManager.default.fileExists(atPath: destination.path))
            let data = try Data(contentsOf: destination)
            if format == "json" {
                XCTAssertNoThrow(try JSONSerialization.jsonObject(with: data))
            } else if format == "vtt" {
                XCTAssertTrue(String(decoding: data, as: UTF8.self).hasPrefix("WEBVTT"))
            }
        }
    }

    func testJSONStdoutRemainsMachineReadableWhileDiagnosticsUseStderr() throws {
        let temporary = try temporaryDirectory()
        let audio = temporary.appendingPathComponent("sample.wav")
        if let fixture = ProcessInfo.processInfo.environment["GRANITE_TEST_AUDIO"] {
            try FileManager.default.copyItem(at: URL(fileURLWithPath: fixture), to: audio)
        } else {
            try writeSilentWAV(to: audio)
        }
        var arguments = [audio.path, "--output-format", "json", "--verbose", "--benchmark"]
        arguments += try realModelArguments()
        let result = try runCLI(arguments, timeout: 180)
        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertNoThrow(try JSONSerialization.jsonObject(with: Data(result.stdout.utf8)))
        XCTAssertTrue(result.stderr.contains("Inference"))
        XCTAssertFalse(result.stdout.contains("Inference"))
        XCTAssertFalse(result.stdout.contains("Download"))
    }

    func testOptInCoreMLBackendMatchesReferenceTranscript() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let model = environment["GRANITE_TEST_COREML_MODEL"],
              let tokenizer = environment["GRANITE_TEST_COREML_TOKENIZER"],
              let audio = environment["GRANITE_TEST_COREML_AUDIO"] else {
            throw XCTSkip(
                "Set GRANITE_TEST_COREML_MODEL, GRANITE_TEST_COREML_TOKENIZER, and GRANITE_TEST_COREML_AUDIO to run Core ML parity.")
        }
        let result = try runCLI([
            audio, "--backend", "coreml", "--coreml-model", model,
            "--model", tokenizer, "--no-punctuate", "--no-chunking",
            "--output-format", "txt",
        ], timeout: 180)
        XCTAssertEqual(result.status, 0, result.stderr)
        XCTAssertEqual(
            result.stdout.trimmingCharacters(in: .whitespacesAndNewlines),
            "hello hello everyone and welcome to cme 295 transformers and large language models so my name is afshin and i will be teaching this class with shervin who is in the back and")
        XCTAssertEqual(result.stderr, "")
    }

    func testOptInPublishedCoreMLFreshDownloadWarmRunAndRemoval() throws {
        let processEnvironment = ProcessInfo.processInfo.environment
        guard processEnvironment["GRANITE_TEST_COREML_NETWORK"] == "1",
              let audio = processEnvironment["GRANITE_TEST_COREML_AUDIO"] else {
            throw XCTSkip(
                "Set GRANITE_TEST_COREML_NETWORK=1 and GRANITE_TEST_COREML_AUDIO to run the published Core ML lifecycle gate.")
        }
        let temporary = try temporaryDirectory()
        let hub = temporary.appendingPathComponent("hub")
        let compiled = temporary.appendingPathComponent("compiled")
        let environment = [
            "GRANITE_MLX_HUB_DIRECTORY": hub.path,
            "GRANITE_MLX_COREML_CACHE_DIRECTORY": compiled.path,
            "GRANITE_MLX_TEST_CONFIG_PATH": temporary
                .appendingPathComponent("config.json").path,
        ]

        let configured = try runCLI(
            ["config", "set", "backend", "coreml"], environment: environment)
        XCTAssertEqual(configured.status, 0, configured.stderr)

        let download = try runCLI(
            ["models", "download", "apache-coreml-q8"],
            environment: environment, timeout: 1_200)
        XCTAssertEqual(download.status, 0, download.stderr)
        XCTAssertTrue(download.stderr.contains("Downloaded:"), download.stderr)

        let arguments = [
            audio, "--no-punctuate", "--no-chunking", "--output-format", "txt",
        ]
        let fresh = try runCLI(arguments, environment: environment, timeout: 300)
        XCTAssertEqual(fresh.status, 0, fresh.stderr)
        let expected = "hello hello everyone and welcome to cme 295 transformers and large language models so my name is afshin and i will be teaching this class with shervin who is in the back and"
        XCTAssertEqual(
            fresh.stdout.trimmingCharacters(in: .whitespacesAndNewlines), expected)

        let warm = try runCLI(arguments, environment: environment, timeout: 300)
        XCTAssertEqual(warm.status, 0, warm.stderr)
        XCTAssertEqual(warm.stdout, fresh.stdout)
        XCTAssertFalse(warm.stderr.contains("Downloading"), warm.stderr)

        let list = try runCLI(["models", "list", "--json"], environment: environment)
        XCTAssertEqual(list.status, 0, list.stderr)
        let records = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(list.stdout.utf8)) as? [[String: Any]])
        let coreML = try XCTUnwrap(records.first { ($0["alias"] as? String) == "apache-coreml-q8" })
        XCTAssertEqual(coreML["cache_state"] as? String, "downloaded")
        XCTAssertGreaterThan(coreML["downloaded_bytes"] as? Int ?? 0, 600_000_000)
        XCTAssertGreaterThan(coreML["compiled_cache_bytes"] as? Int ?? 0, 600_000_000)

        let removal = try runCLI(
            ["models", "remove", "apache-coreml-q8", "--yes"],
            environment: environment)
        XCTAssertEqual(removal.status, 0, removal.stderr)
        XCTAssertTrue(removal.stdout.contains("Reclaimed"), removal.stdout)
        let afterRemoval = try runCLI(
            ["models", "list", "--json"], environment: environment)
        let afterRecords = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(afterRemoval.stdout.utf8)) as? [[String: Any]])
        let afterCoreML = try XCTUnwrap(
            afterRecords.first { ($0["alias"] as? String) == "apache-coreml-q8" })
        XCTAssertEqual(afterCoreML["cache_state"] as? String, "absent")
        XCTAssertEqual(afterCoreML["compiled_cache_bytes"] as? Int ?? 0, 0)
    }

    func testCorruptInputWithoutFFmpegHasCodedError() throws {
        let temporary = try temporaryDirectory()
        let input = temporary.appendingPathComponent("corrupt.webm")
        try Data("not media".utf8).write(to: input)
        var arguments = [input.path, "--no-punctuate"]
        arguments += try realModelArguments().filter { $0 != "--no-punctuate" }
        let result = try runCLI(
            arguments, environment: ["PATH": "/usr/bin:/bin"], timeout: 120)
        XCTAssertNotEqual(result.status, 0)
        XCTAssertEqual(result.stdout, "")
        XCTAssertTrue(result.stderr.contains("GMLX-AUDIO-005"), result.stderr)
        XCTAssertTrue(result.stderr.contains("brew install ffmpeg"), result.stderr)
        XCTAssertTrue(result.stderr.contains("Technical details:"), result.stderr)
    }

    func testCtrlCCancelsWithoutLeavingOutputFiles() throws {
        guard let longAudio = ProcessInfo.processInfo.environment["GRANITE_TEST_LONG_AUDIO"] else {
            throw XCTSkip("Set GRANITE_TEST_LONG_AUDIO to run the Ctrl-C integration test.")
        }
        let temporary = try temporaryDirectory()
        let output = temporary.appendingPathComponent("output")
        var arguments = [
            longAudio, "--no-punctuate", "--output-format", "all",
            "--output-dir", output.path,
        ]
        arguments += try realModelArguments().filter { $0 != "--no-punctuate" }
        let result = try runCLI(
            arguments, timeout: 120, interruptAfter: 0.5)
        XCTAssertNotEqual(result.status, 0)
        XCTAssertTrue(result.stderr.contains("GMLX-OP-001"), result.stderr)
        let files = (try? FileManager.default.contentsOfDirectory(atPath: output.path)) ?? []
        XCTAssertTrue(files.isEmpty, "Cancellation left output files: \(files)")
    }

    func testOptInInterruptedDownloadResumesToCompleteCache() throws {
        guard let model = ProcessInfo.processInfo.environment["GRANITE_TEST_NETWORK_MODEL"] else {
            throw XCTSkip("Set GRANITE_TEST_NETWORK_MODEL to run the network interruption/resume gate.")
        }
        let temporary = try temporaryDirectory()
        let hub = temporary.appendingPathComponent("hub")
        let environment = ["GRANITE_MLX_HUB_DIRECTORY": hub.path]
        let interrupted = try runCLI(
            ["models", "download", model], environment: environment,
            timeout: 120, interruptAfter: 1.0)
        XCTAssertNotEqual(interrupted.status, 0, "Download completed before it could be interrupted; select a larger uncached model.")
        XCTAssertTrue(interrupted.stderr.contains("GMLX-OP-001"), interrupted.stderr)

        let partialList = try runCLI(
            ["models", "list", "--json"], environment: environment)
        XCTAssertEqual(partialList.status, 0)
        let partialRecords = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(partialList.stdout.utf8)) as? [[String: Any]])
        let resolvedAlias = partialRecords.first {
            ($0["alias"] as? String) == model || ($0["repository_id"] as? String) == model
        }
        if let state = resolvedAlias?["cache_state"] as? String {
            XCTAssertTrue(["absent", "partial"].contains(state))
        }

        let resumed = try runCLI(
            ["models", "download", model], environment: environment, timeout: 600)
        XCTAssertEqual(resumed.status, 0, resumed.stderr)
        XCTAssertTrue(
            resumed.stderr.contains("Downloaded:") || resumed.stderr.contains("Already downloaded:"),
            resumed.stderr)
        let completeList = try runCLI(
            ["models", "list", "--json"], environment: environment)
        let completeRecords = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(completeList.stdout.utf8)) as? [[String: Any]])
        let completed = completeRecords.first {
            ($0["alias"] as? String) == model || ($0["repository_id"] as? String) == model
        }
        XCTAssertEqual(completed?["cache_state"] as? String, "downloaded")
    }

    func testOptInLongFormBoundedMemoryRegression() throws {
        guard let audio = ProcessInfo.processInfo.environment["GRANITE_TEST_LONG_AUDIO"] else {
            throw XCTSkip("Set GRANITE_TEST_LONG_AUDIO to run the long-form release gate.")
        }
        let temporary = try temporaryDirectory()
        let output = temporary.appendingPathComponent("long.json")
        var arguments = [
            audio, "--no-punctuate", "--output-format", "json",
            "--output-dir", temporary.path, "--output-template", "long",
            "--mlx-cache-limit-mb", "64",
        ]
        arguments += try realModelArguments().filter { $0 != "--no-punctuate" }
        let result = try runCLI(arguments, timeout: 600)
        XCTAssertEqual(result.status, 0, result.stderr)
        let document = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: output)) as? [String: Any])
        let text = try XCTUnwrap(document["text"] as? String)
        XCTAssertFalse(text.isEmpty)
        let performance = try XCTUnwrap(document["performance"] as? [String: Any])
        XCTAssertEqual(performance["audio_chunk_duration_seconds"] as? Double, 122.88)
        XCTAssertEqual(performance["audio_chunk_context_seconds"] as? Double, 10.24)
        let peak = try XCTUnwrap(performance["mlx_peak_memory_bytes"] as? Int)
        let maximum = Int(ProcessInfo.processInfo.environment["GRANITE_TEST_MAX_MLX_PEAK_BYTES"] ?? "2500000000")!
        XCTAssertLessThan(peak, maximum, "MLX peak memory exceeded the release threshold")
        if let expectedPath = ProcessInfo.processInfo.environment["GRANITE_TEST_LONG_EXPECTED_TEXT"] {
            let expected = try String(contentsOfFile: expectedPath, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if text != expected {
                let mismatch = zip(text, expected).enumerated().first {
                    $0.element.0 != $0.element.1
                }?.offset ?? min(text.count, expected.count)
                XCTFail(
                    "Long-form transcript differs from its selected baseline at character \(mismatch); "
                    + "actual_chars=\(text.count), expected_chars=\(expected.count). "
                    + "Make sure GRANITE_TEST_SPEECH_MODEL matches the checkpoint used by the baseline.")
            }
        }
    }

    private func writeSafetensorsHeader(names: [String], to url: URL) throws {
        let object = Dictionary(uniqueKeysWithValues: names.map { ($0, [String: Any]()) })
        let header = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        var length = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(header)
        try data.write(to: url)
    }
}
