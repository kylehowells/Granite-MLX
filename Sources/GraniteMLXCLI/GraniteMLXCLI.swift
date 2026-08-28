import ArgumentParser
import Darwin
import Foundation
import GraniteMLX
import MLX

func stderr(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

final class CLIInterruptHandler {
    let cancellationToken = GraniteCancellationToken()
    private let source: DispatchSourceSignal

    init() {
        signal(SIGINT, SIG_IGN)
        source = DispatchSource.makeSignalSource(
            signal: SIGINT, queue: .global(qos: .userInitiated))
        source.setEventHandler { [cancellationToken] in
            if cancellationToken.isCancelled {
                stderr("[GMLX-OP-001] Cancellation is already in progress; waiting for the current MLX operation to yield.")
            } else {
                stderr("[GMLX-OP-001] Cancellation requested; cleaning up the current operation.")
                cancellationToken.cancel()
            }
        }
        source.activate()
    }

    deinit {
        source.cancel()
        signal(SIGINT, SIG_DFL)
    }
}

struct CLIRuntimeError: Error, LocalizedError {
    let code: String
    let message: String
    let details: String

    var errorDescription: String? {
        "[\(code)] \(message) Technical details: \(details)"
    }

    static func wrapping(_ error: Error, code: String, operation: String) -> CLIRuntimeError {
        if let diagnostic = error as? any GraniteDiagnosticError {
            let prefix = "[\(diagnostic.diagnosticCode)] "
            let localized = diagnostic.errorDescription ?? "\(operation) failed."
            let withoutPrefix = localized.hasPrefix(prefix) ? String(localized.dropFirst(prefix.count)) : localized
            let concise = withoutPrefix.components(separatedBy: " Technical details:").first ?? withoutPrefix
            return CLIRuntimeError(
                code: diagnostic.diagnosticCode,
                message: concise,
                details: diagnostic.technicalDetails ?? String(reflecting: error))
        }
        return CLIRuntimeError(
            code: code,
            message: "\(operation) failed.",
            details: "error_type=\(String(reflecting: type(of: error))); underlying=\(String(reflecting: error))")
    }
}

private enum OutputFormat: String, CaseIterable {
    case txt, srt, vtt, json, all
}

private struct BenchmarkResult: Encodable {
    let audioFile: String
    let backend: String
    let model: String
    let audioDurationSeconds: Double
    let audioLoadSeconds: Double
    let modelLoadSeconds: Double
    let inferenceSeconds: Double
    let punctuationModel: String?
    let punctuationModelLoadSeconds: Double
    let punctuationInferenceSeconds: Double
    let totalSeconds: Double
    let realTimeFactor: Double
    let realtimeMultiple: Double
    let activationPrecision: String
    let ctcVocabularyTileSize: Int
    let middleCTCVocabularyTileSize: Int
    let audioChunkDurationSeconds: Double
    let audioChunkContextSeconds: Double
    let mlxActiveMemoryBytes: Int
    let mlxCacheMemoryBytes: Int
    let mlxPeakMemoryBytes: Int
    let mlxCacheLimitBytes: Int
    let coremlChunkCount: Int?
    let coremlFeatureExtractionSeconds: Double?
    let coremlInputCopySeconds: Double?
    let coremlPredictionSeconds: Double?
    let coremlPredictionDurations: [Double]?
    let coremlOutputCopySeconds: Double?
}

private struct TranscriptionDocument: Encodable {
    let audioFile: String
    let backend: String
    let text: String
    let rawText: String
    let formattedText: String?
    let durationSeconds: Double
    let model: String
    let weightQuantizationBits: Int?
    let punctuationModel: String?
    let punctuationPrecision: String?
    let punctuationQuantizationBits: Int?
    let activationPrecision: String
    let timingNote: String
    let words: [GraniteWord]
    let segments: [GraniteSubtitleSegment]
    let performance: BenchmarkResult
}

private final class OutputPathResolver {
    private var reserved: Set<String> = []

    func destination(directory: URL, baseName: String, extension ext: String) -> URL {
        var suffix = 1
        while true {
            let name = suffix == 1 ? baseName : "\(baseName)-\(suffix)"
            let candidate = directory.appendingPathComponent(name).appendingPathExtension(ext)
            let path = candidate.standardizedFileURL.path
            if !reserved.contains(path), !FileManager.default.fileExists(atPath: path) {
                reserved.insert(path)
                return candidate
            }
            suffix += 1
        }
    }
}

@main
struct GraniteMLXCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "granite-mlx",
        abstract: "Run Granite Speech 5.0 locally on audio or video files.",
        discussion: """
        COMMON EXAMPLES:
          granite-mlx recording.wav
          granite-mlx lecture.mp4 --output-format vtt
          granite-mlx interview.m4a --output-format all --output-dir ./transcripts
          granite-mlx recording.wav --no-punctuate --output-format txt
          granite-mlx recording.wav --backend coreml
          granite-mlx config set backend coreml
          granite-mlx models list
          granite-mlx models download apache-coreml-q8 punctuation-q8

        `transcribe` is the default command, so `granite-mlx recording.wav` and
        `granite-mlx transcribe recording.wav` are equivalent.

        MLX is the built-in speech backend. Use --backend coreml for one run, or
        `granite-mlx config set backend coreml` to save it as your default.
        The published Core ML model requires macOS 15 or newer.
        """,
        version: "0.1.1",
        subcommands: [TranscribeCommand.self, ModelsCommand.self, ConfigCommand.self],
        defaultSubcommand: TranscribeCommand.self
    )
}

struct TranscribeCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe audio or video with Granite Speech 5.0.",
        discussion: """
        EXAMPLES:
          granite-mlx recording.wav
          granite-mlx lecture.mp4 --output-format srt
          granite-mlx lecture.mp4 --output-format all --output-dir ./transcripts
          granite-mlx a.wav b.mp3 --output-dir ./transcripts
          granite-mlx recording.wav --model apache-q6 --no-punctuate
          granite-mlx recording.wav --backend coreml
          granite-mlx recording.wav --backend coreml --model apache-coreml-q8
          granite-mlx a.wav --backend coreml --coreml-model ./repo/G.mlpackage

        SRT is written to stdout by default. Use --output-format txt for plain
        text, or --output-dir to create files. The first run downloads and
        caches the selected models with progress on stderr.

        MLX uses apache-q8 by default; Core ML uses apache-coreml-q8. Run
        `granite-mlx config set backend coreml` to save Core ML as your default.
        An explicit --backend always wins. The published Core ML model requires
        macOS 15 or newer.
        """)

    @Argument(help: "Audio or video file(s) to transcribe.")
    var inputs: [String]
    @Option(help: "Backend-compatible catalog alias, local repository directory, or Hugging Face repository ID. Defaults to apache-q8 for MLX or apache-coreml-q8 for Core ML.")
    var model: String?
    @Option(help: "Speech backend: mlx or coreml. Overrides the saved user setting for this run.")
    var backend: String?
    @Option(help: "Advanced: local fixed-shape .mlpackage override for Core ML. Use --model for its tokenizer directory.")
    var coremlModel: String?
    @Option(help: "Core ML compute units: cpu-gpu, all, cpu-ne, or cpu.")
    var coremlComputeUnits: String = GraniteCoreMLComputeUnits.cpuAndGPU.rawValue
    @Option(help: "Catalog alias, local punctuation model directory, or Hugging Face repository ID.")
    var punctuationModel: String = PunctuationModelLoader.defaultModelID
    @Option(help: "Hugging Face access token for private or gated repositories; overrides HF_TOKEN.")
    var hfToken: String?
    @Flag(inversion: .prefixedNo, help: "Restore punctuation, capitalization, and sentence boundaries.")
    var punctuate = true
    @Option(help: "Output format: txt, srt, vtt, json, or all; all requires --output-dir.")
    var outputFormat: String = "srt"
    @Option(help: "Directory for generated files; without it, output is written to stdout.")
    var outputDir: String?
    @Option(help: "Output basename template. Supports {filename}, {parent}, {index}, and {date}.")
    var outputTemplate: String = "{filename}"
    @Flag(help: "Emit one subtitle cue per word and highlight the active word.")
    var highlightWords = false
    @Option(help: "Maximum words per subtitle cue.")
    var maxWords: Int = 20
    @Option(help: "Start a subtitle cue after this many seconds between word onsets.")
    var silenceGap: Double = 1
    @Option(help: "Maximum subtitle cue duration in seconds.")
    var maxDuration: Double = 8
    @Flag(help: "Print diagnostics and performance information.")
    var verbose = false
    @Flag(help: "Write machine-readable timing results to stderr.")
    var benchmark = false
    @Option(help: "MLX-only encoder activations: baseline, fp16, fp8-emulated, or int8-emulated.")
    var activationPrecision: String = GraniteActivationPrecision.fp16.rawValue
    @Option(help: "MLX-only: stream final CTC argmax in vocabulary tiles; 0 materializes all logits.")
    var ctcVocabularyTile: Int = 0
    @Option(help: "MLX-only: stream middle CTC softmax/projection in vocabulary tiles; 0 materializes all logits.")
    var middleCTCVocabularyTile: Int = 0
    @Option(help: "Limit MLX's recycled-buffer cache in MiB.")
    var mlxCacheLimitMB: Int = 64
    @Option(help: "Central audio seconds per chunk. Defaults to 122.88 for MLX or the largest input that fits the selected Core ML graph. Multiples of 10.24s preserve attention-block alignment.")
    var audioChunkDuration: Double?
    @Option(help: "Extra context on each side of a chunk. Defaults to 10.24s for MLX or up to 20.48s for Core ML.")
    var audioChunkContext: Double?
    @Flag(help: "Disable temporal chunking; the input must fit the selected backend model's one-pass limit.")
    var noChunking = false
    @Option(help: "Diagnostic: write the first input's frontend tensor as safetensors and exit.")
    var dumpFeatures: String?
    @Option(help: "Diagnostic: write activation shapes, dtypes, and ranges as JSON and exit.")
    var activationAudit: String?

    func validate() throws {
        guard !inputs.isEmpty else { throw ValidationError("[GMLX-CLI-001] At least one input file is required.") }
        guard OutputFormat(rawValue: outputFormat.lowercased()) != nil else {
            throw ValidationError("[GMLX-CLI-002] Unsupported output format `\(outputFormat)`. Use txt, srt, vtt, json, or all.")
        }
        if outputFormat.lowercased() == "all", outputDir == nil {
            throw ValidationError("[GMLX-CLI-003] --output-format all requires --output-dir.")
        }
        guard maxWords > 0 else { throw ValidationError("[GMLX-CLI-004] --max-words must be greater than zero; received \(maxWords).") }
        guard silenceGap >= 0 else { throw ValidationError("[GMLX-CLI-005] --silence-gap must be non-negative; received \(silenceGap).") }
        guard maxDuration > 0 else { throw ValidationError("[GMLX-CLI-006] --max-duration must be greater than zero; received \(maxDuration).") }
        guard mlxCacheLimitMB >= 0 else { throw ValidationError("[GMLX-CLI-007] --mlx-cache-limit-mb must be non-negative; received \(mlxCacheLimitMB).") }
        guard (noChunking ? 0 : (audioChunkDuration ?? 0)) >= 0,
              (noChunking ? 0 : (audioChunkContext ?? 0)) >= 0 else {
            throw ValidationError("[GMLX-CLI-008] Audio chunk duration and context must be non-negative; duration=\(String(describing: audioChunkDuration)), context=\(String(describing: audioChunkContext)).")
        }
        guard GraniteActivationPrecision(rawValue: activationPrecision) != nil else {
            throw ValidationError("[GMLX-CLI-009] Unsupported activation precision `\(activationPrecision)`. Use baseline, fp16, fp8-emulated, or int8-emulated.")
        }
        let selectedBackend = try resolvedBackend()
        if selectedBackend == .coreML {
            guard #available(macOS 15.0, *) else {
                throw ValidationError(
                    "[GMLX-CLI-018] The published Core ML backend requires macOS 15 or newer. "
                    + "Use --backend mlx on this Mac.")
            }
        }
        if selectedBackend == .mlx, coremlModel != nil {
            throw ValidationError("[GMLX-CLI-014] --coreml-model requires the Core ML backend. Pass --backend coreml or save it with `granite-mlx config set backend coreml`.")
        }
        if let model,
           let catalogModel = GraniteModelCatalog.models.first(where: {
               $0.alias.caseInsensitiveCompare(model) == .orderedSame
                   || $0.repositoryID.caseInsensitiveCompare(model) == .orderedSame
           }) {
            let expectedKind: GraniteManagedModelKind = selectedBackend == .coreML
                ? .coreMLSpeech : .speech
            guard catalogModel.kind == expectedKind else {
                throw ValidationError(
                    "[GMLX-CLI-017] Model `\(model)` is a \(catalogModel.kind.rawValue) model "
                    + "and cannot run with backend `\(selectedBackend.rawValue)`. "
                    + "Use --backend \(catalogModel.kind == .coreMLSpeech ? "coreml" : "mlx") "
                    + "or select a backend-compatible speech model.")
            }
        }
        guard GraniteCoreMLComputeUnits(rawValue: coremlComputeUnits.lowercased()) != nil else {
            throw ValidationError("[GMLX-CLI-015] Unsupported Core ML compute-unit policy `\(coremlComputeUnits)`. Use cpu-gpu, all, cpu-ne, or cpu.")
        }
    }

    func run() throws {
        let interruptHandler = CLIInterruptHandler()
        let cancellationToken = interruptHandler.cancellationToken
        let format = OutputFormat(rawValue: outputFormat.lowercased())!
        let precision = GraniteActivationPrecision(rawValue: activationPrecision)!
        let selectedBackend = try resolvedBackend()
        let selectedModel = model ?? (selectedBackend == .coreML
            ? GraniteCoreMLModelLoader.defaultModelID
            : GraniteModelLoader.defaultModelID)
        let outputDirectory: URL?
        do { outputDirectory = try prepareOutputDirectory() }
        catch {
            throw CLIRuntimeError.wrapping(
                error, code: "GMLX-CLI-010", operation: "Output directory preparation")
        }
        let resolver = OutputPathResolver()
        let effectiveHFToken = hfToken ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
        let modelStorage = GraniteModelStorage.default

        Memory.cacheLimit = mlxCacheLimitMB * 1_024 * 1_024
        Memory.clearCache()
        let downloadReporter = ConsoleDownloadProgressReporter()
        let modelStart = Date()
        let mlxRecognizer: GraniteRecognizer?
        let coreMLRecognizer: GraniteCoreMLRecognizer?
        let coreMLArtifact: GraniteCoreMLModelArtifact?
        if selectedBackend == .coreML {
            mlxRecognizer = nil
            do {
                if let coremlModel {
                    coreMLArtifact = nil
                    let packageURL = URL(fileURLWithPath: coremlModel)
                        .standardizedFileURL
                    let tokenizerURL: URL
                    if let model {
                        tokenizerURL = URL(fileURLWithPath: model)
                            .standardizedFileURL
                    } else {
                        tokenizerURL = packageURL.deletingLastPathComponent()
                    }
                    coreMLRecognizer = try GraniteCoreMLRecognizer(
                        modelURL: packageURL,
                        tokenizerURL: tokenizerURL,
                        computeUnits: GraniteCoreMLComputeUnits(
                            rawValue: coremlComputeUnits.lowercased())!)
                } else {
                    let artifact = try GraniteCoreMLModelLoader.load(
                        source: selectedModel,
                        storage: modelStorage,
                        hfToken: effectiveHFToken,
                        progressHandler: downloadReporter.handler,
                        cancellationToken: cancellationToken)
                    coreMLArtifact = artifact
                    coreMLRecognizer = try GraniteCoreMLRecognizer(
                        artifact: artifact,
                        computeUnits: GraniteCoreMLComputeUnits(
                            rawValue: coremlComputeUnits.lowercased())!)
                }
            } catch {
                throw CLIRuntimeError.wrapping(
                    error, code: "GMLX-CLI-020", operation: "Core ML speech model initialization")
            }
        } else {
            coreMLRecognizer = nil
            coreMLArtifact = nil
            do {
                mlxRecognizer = try GraniteRecognizer(
                    modelSource: selectedModel, storage: modelStorage,
                    hfToken: effectiveHFToken,
                    progressHandler: downloadReporter.handler,
                    cancellationToken: cancellationToken)
            } catch {
                throw CLIRuntimeError.wrapping(
                    error, code: "GMLX-CLI-020", operation: "MLX speech model initialization")
            }
        }
        let modelLoadSeconds = Date().timeIntervalSince(modelStart)
        let chunkContext: Double
        if noChunking {
            chunkContext = 0
        } else if let audioChunkContext {
            chunkContext = audioChunkContext
        } else if let coreMLRecognizer {
            chunkContext = min(20.48, coreMLRecognizer.maximumAudioDuration / 4)
        } else {
            chunkContext = GraniteRecognizer.defaultAudioChunkContext
        }
        let chunkDuration: Double
        if noChunking {
            chunkDuration = 0
        } else if let audioChunkDuration {
            chunkDuration = audioChunkDuration
        } else if let coreMLRecognizer {
            chunkDuration = max(
                0, coreMLRecognizer.maximumAudioDuration - 2 * chunkContext)
        } else {
            chunkDuration = GraniteRecognizer.defaultAudioChunkDuration
        }
        let speechModelDescription = coremlModel ?? selectedModel
        let speechWeightBits = mlxRecognizer?.artifact.configuration.quantization?.bits
            ?? coreMLArtifact?.configuration.quantization.bits
        let runtimePrecision = coreMLRecognizer == nil ? precision.rawValue : "coreml-fp16"

        let formatter: (any GraniteTranscriptFormatter)?
        let punctuationLoadSeconds: Double
        if punctuate {
            let start = Date()
            do {
                formatter = try GraniteTranscriptFormatterFactory.load(
                    modelSource: punctuationModel, storage: modelStorage,
                    hfToken: effectiveHFToken,
                    progressHandler: downloadReporter.handler,
                    cancellationToken: cancellationToken)
            } catch {
                throw CLIRuntimeError.wrapping(
                    error, code: "GMLX-CLI-021", operation: "Punctuation model initialization")
            }
            punctuationLoadSeconds = Date().timeIntervalSince(start)
        } else {
            formatter = nil
            punctuationLoadSeconds = 0
        }

        if let dumpFeatures {
            do {
                let audio = try GraniteAudioInput.load(
                    url: URL(fileURLWithPath: inputs[0]),
                    cancellationToken: cancellationToken)
                let features = mlxRecognizer?.features(for: audio)
                    ?? coreMLRecognizer!.features(for: audio)
                MLX.eval(features)
                try MLX.save(arrays: ["features": features], metadata: [:], url: URL(fileURLWithPath: dumpFeatures))
                if verbose { stderr("Wrote frontend tensor \(features.shape) to \(dumpFeatures)") }
            } catch {
                throw CLIRuntimeError.wrapping(
                    error, code: "GMLX-CLI-011", operation: "Frontend tensor export")
            }
            return
        }
        if let activationAudit {
            guard let mlxRecognizer else {
                throw ValidationError("[GMLX-CLI-016] --activation-audit is currently available only with --backend mlx.")
            }
            do {
                let audio = try GraniteAudioInput.load(
                    url: URL(fileURLWithPath: inputs[0]),
                    cancellationToken: cancellationToken)
                let stages = mlxRecognizer.activationAudit(audio, activationPrecision: precision)
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(stages).write(to: URL(fileURLWithPath: activationAudit))
                if verbose { stderr("Wrote activation audit to \(activationAudit)") }
            } catch {
                throw CLIRuntimeError.wrapping(
                    error, code: "GMLX-CLI-012", operation: "Activation audit export")
            }
            return
        }

        if verbose, inputs.count > 1 {
            let loadedModels = punctuate ? "ASR and punctuation models" : "ASR model"
            stderr("Loaded \(loadedModels) once; processing \(inputs.count) inputs.")
            if outputDirectory == nil { stderr("Multiple stdout results are emitted as JSON Lines.") }
        }

        var failedInputs = 0
        for (index, input) in inputs.enumerated() {
            do {
            let totalStart = Date()
            let inputURL = URL(fileURLWithPath: input).standardizedFileURL
            let audioStart = Date()
            let audio = try GraniteAudioInput.load(
                url: inputURL, cancellationToken: cancellationToken)
            let audioLoadSeconds = Date().timeIntervalSince(audioStart)
            if verbose {
                stderr(String(format: "Loaded %@: %.2fs at %d Hz", inputURL.lastPathComponent, audio.duration, audio.sampleRate))
            }

            Memory.peakMemory = 0
            let inferenceStart = Date()
            let rawResult: GraniteTranscription
            if let coreMLRecognizer {
                rawResult = try coreMLRecognizer.transcribe(
                    audio,
                    audioChunkDuration: chunkDuration,
                    audioChunkContext: chunkContext,
                    cancellationToken: cancellationToken)
            } else {
                rawResult = try mlxRecognizer!.transcribe(
                    audio,
                    activationPrecision: precision,
                    ctcVocabularyTileSize: ctcVocabularyTile,
                    middleCTCVocabularyTileSize: middleCTCVocabularyTile,
                    audioChunkDuration: chunkDuration,
                    audioChunkContext: chunkContext,
                    cancellationToken: cancellationToken)
            }
            let inferenceSeconds = Date().timeIntervalSince(inferenceStart)

            let punctuationStart = Date()
            let formatting = try formatter?.format(
                rawResult.rawText, cancellationToken: cancellationToken,
                progressHandler: nil)
            let result = formatting.map(rawResult.applyingFormatting) ?? rawResult
            let punctuationInferenceSeconds = Date().timeIntervalSince(punctuationStart)
            if formatting != nil, result.words.count != rawResult.words.count, verbose {
                stderr("Warning: formatter changed the word count; subtitle words retain raw text.")
            }
            let segments = GraniteSubtitleSegmenter.segments(
                words: result.words,
                sentenceWordRanges: formatting?.sentenceWordRanges ?? [],
                maxWords: maxWords,
                silenceGap: silenceGap,
                maxDuration: maxDuration)
            let memory = Memory.snapshot()
            let coreMLPerformance = coreMLRecognizer?.lastPerformance
            let report = BenchmarkResult(
                audioFile: inputURL.path,
                backend: selectedBackend.rawValue,
                model: speechModelDescription,
                audioDurationSeconds: audio.duration,
                audioLoadSeconds: audioLoadSeconds,
                modelLoadSeconds: modelLoadSeconds,
                inferenceSeconds: inferenceSeconds,
                punctuationModel: punctuate ? punctuationModel : nil,
                punctuationModelLoadSeconds: punctuationLoadSeconds,
                punctuationInferenceSeconds: punctuationInferenceSeconds,
                totalSeconds: Date().timeIntervalSince(totalStart),
                realTimeFactor: (inferenceSeconds + punctuationInferenceSeconds) / max(audio.duration, 1e-9),
                realtimeMultiple: audio.duration / max(inferenceSeconds + punctuationInferenceSeconds, 1e-9),
                activationPrecision: runtimePrecision,
                ctcVocabularyTileSize: ctcVocabularyTile,
                middleCTCVocabularyTileSize: middleCTCVocabularyTile,
                audioChunkDurationSeconds: chunkDuration,
                audioChunkContextSeconds: chunkContext,
                mlxActiveMemoryBytes: memory.activeMemory,
                mlxCacheMemoryBytes: memory.cacheMemory,
                mlxPeakMemoryBytes: memory.peakMemory,
                mlxCacheLimitBytes: Memory.cacheLimit,
                coremlChunkCount: coreMLPerformance?.chunkCount,
                coremlFeatureExtractionSeconds: coreMLPerformance?.featureExtractionSeconds,
                coremlInputCopySeconds: coreMLPerformance?.inputCopySeconds,
                coremlPredictionSeconds: coreMLPerformance?.predictionSeconds,
                coremlPredictionDurations: coreMLPerformance?.predictionDurations,
                coremlOutputCopySeconds: coreMLPerformance?.outputCopySeconds)
            let document = TranscriptionDocument(
                audioFile: inputURL.path,
                backend: selectedBackend.rawValue,
                text: result.text,
                rawText: result.rawText,
                formattedText: result.formattedText,
                durationSeconds: result.duration,
                model: speechModelDescription,
                weightQuantizationBits: speechWeightBits,
                punctuationModel: punctuate ? punctuationModel : nil,
                punctuationPrecision: formatter?.formatterInfo.precision,
                punctuationQuantizationBits: formatter?.formatterInfo.quantizationBits,
                activationPrecision: runtimePrecision,
                timingNote: "Word timings are approximate CTC emission-frame alignments.",
                words: result.words,
                segments: segments,
                performance: report)

            try emit(
                format: format, document: document, transcription: result,
                segments: segments, inputURL: inputURL, inputIndex: index,
                outputDirectory: outputDirectory, resolver: resolver,
                forceJSONLine: outputDirectory == nil && inputs.count > 1,
                cancellationToken: cancellationToken)
            if verbose {
                stderr(String(format: "Inference %.3fs; formatting %.3fs; %.1fx realtime", inferenceSeconds, punctuationInferenceSeconds, report.realtimeMultiple))
            }
            if benchmark { stderr(try encodedJSON(report, pretty: false)) }
            Memory.clearCache()
            } catch let error as GraniteOperationError {
                if case .cancelled = error { throw error }
                failedInputs += 1
                stderr(CLIRuntimeError.wrapping(
                    error, code: "GMLX-CLI-030", operation: "Processing input \(input)").localizedDescription)
                Memory.clearCache()
            } catch {
                failedInputs += 1
                stderr(CLIRuntimeError.wrapping(
                    error, code: "GMLX-CLI-030", operation: "Processing input \(input)").localizedDescription)
                Memory.clearCache()
            }
        }
        if failedInputs > 0 {
            stderr("[GMLX-CLI-031] Completed batch with \(failedInputs) failed input(s) out of \(inputs.count). Successfully generated outputs were retained. Technical details: failures=\(failedInputs); total=\(inputs.count)")
            throw ExitCode.failure
        }
    }

    private func resolvedBackend() throws -> GraniteSpeechBackend {
        // An explicit invocation must remain usable even when the optional
        // saved configuration is corrupt, because the command-line value has
        // documented precedence over that file.
        let value: String
        if let backend {
            value = backend
        } else {
            value = try CLIConfigurationStore.load().defaultBackend?.rawValue
                ?? GraniteSpeechBackend.mlx.rawValue
        }
        guard let backend = GraniteSpeechBackend(rawValue: value.lowercased()) else {
            throw ValidationError(
                "[GMLX-CLI-013] Unsupported speech backend `\(value)`. Use mlx or coreml.")
        }
        return backend
    }

    private func prepareOutputDirectory() throws -> URL? {
        guard let outputDir else { return nil }
        let url = URL(fileURLWithPath: outputDir).standardizedFileURL
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func emit(
        format: OutputFormat,
        document: TranscriptionDocument,
        transcription: GraniteTranscription,
        segments: [GraniteSubtitleSegment],
        inputURL: URL,
        inputIndex: Int,
        outputDirectory: URL?,
        resolver: OutputPathResolver,
        forceJSONLine: Bool,
        cancellationToken: GraniteCancellationToken
    ) throws {
        try cancellationToken.checkCancellation(operation: "Transcript output")
        if forceJSONLine {
            FileHandle.standardOutput.write(Data((try encodedJSON(document, pretty: false) + "\n").utf8))
            return
        }
        let formats: [OutputFormat] = format == .all ? [.txt, .srt, .vtt, .json] : [format]
        var writtenFiles: [URL] = []
        do {
            for selected in formats {
                try cancellationToken.checkCancellation(operation: "Transcript output")
                let content = try rendered(
                    selected, document: document, transcription: transcription,
                    segments: segments)
                if let outputDirectory {
                    let destination = resolver.destination(
                        directory: outputDirectory,
                        baseName: renderedBaseName(for: inputURL, index: inputIndex),
                        extension: selected.rawValue)
                    try Data(content.utf8).write(to: destination, options: .atomic)
                    writtenFiles.append(destination)
                    if verbose { stderr("Wrote \(destination.path)") }
                } else {
                    FileHandle.standardOutput.write(Data(content.utf8))
                }
            }
        } catch {
            for file in writtenFiles { try? FileManager.default.removeItem(at: file) }
            throw error
        }
    }

    private func rendered(
        _ format: OutputFormat,
        document: TranscriptionDocument,
        transcription: GraniteTranscription,
        segments: [GraniteSubtitleSegment]
    ) throws -> String {
        switch format {
        case .txt:
            GraniteTranscriptExporter.text(transcription)
        case .srt:
            GraniteTranscriptExporter.srt(segments: segments, duration: transcription.duration, highlightWords: highlightWords)
        case .vtt:
            GraniteTranscriptExporter.webVTT(segments: segments, duration: transcription.duration, highlightWords: highlightWords)
        case .json:
            try encodedJSON(document, pretty: true) + "\n"
        case .all:
            preconditionFailure("The all format must be expanded before rendering.")
        }
    }

    private func renderedBaseName(for input: URL, index: Int) -> String {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyyMMdd"
        let sourceName = input.deletingPathExtension().lastPathComponent
        let rendered = outputTemplate
            .replacingOccurrences(of: "{filename}", with: sourceName)
            .replacingOccurrences(of: "{parent}", with: input.deletingLastPathComponent().lastPathComponent)
            .replacingOccurrences(of: "{index}", with: String(index))
            .replacingOccurrences(of: "{date}", with: dateFormatter.string(from: Date()))
        let safe = rendered.replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: ":", with: "-")
        return safe.isEmpty ? sourceName : safe
    }
}

private func encodedJSON<T: Encodable>(_ value: T, pretty: Bool) throws -> String {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    encoder.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]
    return String(decoding: try encoder.encode(value), as: UTF8.self)
}
