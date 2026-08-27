import ArgumentParser
import Foundation
import GraniteMLX
import MLX

func stderr(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private enum OutputFormat: String, CaseIterable {
    case txt, srt, vtt, json, all
}

private struct BenchmarkResult: Encodable {
    let audioFile: String
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
}

private struct TranscriptionDocument: Encodable {
    let audioFile: String
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
        version: "0.1.0",
        subcommands: [TranscribeCommand.self, ModelsCommand.self],
        defaultSubcommand: TranscribeCommand.self
    )
}

struct TranscribeCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe audio or video with Granite Speech 5.0.")

    @Argument(help: "Audio or video file(s) to transcribe.")
    var inputs: [String]
    @Option(help: "Local model directory or Hugging Face model identifier.")
    var model: String = GraniteModelLoader.defaultModelID
    @Option(help: "Local punctuation model directory or Hugging Face model identifier.")
    var punctuationModel: String = PunctuationModelLoader.defaultModelID
    @Option(help: "Hugging Face access token for private or gated model repositories.")
    var hfToken: String?
    @Flag(inversion: .prefixedNo, help: "Restore punctuation, capitalization, and sentence boundaries.")
    var punctuate = true
    @Option(help: "Output format: txt, srt, vtt, json, or all.")
    var outputFormat: String = "srt"
    @Option(help: "Directory for generated output files.")
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
    @Option(help: "Encoder activations: baseline, fp16, fp8-emulated, or int8-emulated.")
    var activationPrecision: String = GraniteActivationPrecision.fp16.rawValue
    @Option(help: "Stream final CTC argmax in vocabulary tiles; 0 materializes all logits.")
    var ctcVocabularyTile: Int = 0
    @Option(help: "Stream middle CTC softmax/projection in vocabulary tiles; 0 materializes all logits.")
    var middleCTCVocabularyTile: Int = 0
    @Option(help: "Limit MLX's recycled-buffer cache in MiB.")
    var mlxCacheLimitMB: Int = 64
    @Option(help: "Bound encoder memory with independent audio chunks; 0 runs one pass. Multiples of 10.24s preserve attention-block alignment.")
    var audioChunkDuration: Double = 122.88
    @Option(help: "Extra context on each side of an audio chunk; only central emissions are retained.")
    var audioChunkContext: Double = 20.48
    @Flag(help: "Disable temporal chunking and run the complete recording in one encoder pass.")
    var noChunking = false
    @Option(help: "Diagnostic: write the first input's frontend tensor as safetensors and exit.")
    var dumpFeatures: String?
    @Option(help: "Diagnostic: write activation shapes, dtypes, and ranges as JSON and exit.")
    var activationAudit: String?

    func validate() throws {
        guard !inputs.isEmpty else { throw ValidationError("At least one input file is required.") }
        guard OutputFormat(rawValue: outputFormat.lowercased()) != nil else {
            throw ValidationError("Unsupported output format. Use txt, srt, vtt, json, or all.")
        }
        if outputFormat.lowercased() == "all", outputDir == nil {
            throw ValidationError("--output-format all requires --output-dir.")
        }
        guard maxWords > 0 else { throw ValidationError("--max-words must be greater than zero.") }
        guard silenceGap >= 0 else { throw ValidationError("--silence-gap must be non-negative.") }
        guard maxDuration > 0 else { throw ValidationError("--max-duration must be greater than zero.") }
        guard mlxCacheLimitMB >= 0 else { throw ValidationError("--mlx-cache-limit-mb must be non-negative.") }
        guard (noChunking ? 0 : audioChunkDuration) >= 0,
              (noChunking ? 0 : audioChunkContext) >= 0 else {
            throw ValidationError("Audio chunk duration and context must be non-negative.")
        }
        guard GraniteActivationPrecision(rawValue: activationPrecision) != nil else {
            throw ValidationError("Unsupported activation precision. Use baseline, fp16, fp8-emulated, or int8-emulated.")
        }
    }

    func run() throws {
        let format = OutputFormat(rawValue: outputFormat.lowercased())!
        let precision = GraniteActivationPrecision(rawValue: activationPrecision)!
        let chunkDuration = noChunking ? 0 : audioChunkDuration
        let chunkContext = noChunking ? 0 : audioChunkContext
        let outputDirectory = try prepareOutputDirectory()
        let resolver = OutputPathResolver()

        Memory.cacheLimit = mlxCacheLimitMB * 1_024 * 1_024
        Memory.clearCache()
        let downloadReporter = ConsoleDownloadProgressReporter()
        let modelStart = Date()
        let recognizer = try GraniteRecognizer(
            modelSource: model, hfToken: hfToken,
            progressHandler: downloadReporter.handler)
        let modelLoadSeconds = Date().timeIntervalSince(modelStart)

        let formatter: PunctuationFormatter?
        let punctuationLoadSeconds: Double
        if punctuate {
            let start = Date()
            formatter = try PunctuationFormatter(
                modelSource: punctuationModel, hfToken: hfToken,
                progressHandler: downloadReporter.handler)
            punctuationLoadSeconds = Date().timeIntervalSince(start)
        } else {
            formatter = nil
            punctuationLoadSeconds = 0
        }

        if let dumpFeatures {
            let audio = try GraniteAudioInput.load(url: URL(fileURLWithPath: inputs[0]))
            let features = recognizer.features(for: audio)
            MLX.eval(features)
            try MLX.save(arrays: ["features": features], metadata: [:], url: URL(fileURLWithPath: dumpFeatures))
            if verbose { stderr("Wrote frontend tensor \(features.shape) to \(dumpFeatures)") }
            return
        }
        if let activationAudit {
            let audio = try GraniteAudioInput.load(url: URL(fileURLWithPath: inputs[0]))
            let stages = recognizer.activationAudit(audio, activationPrecision: precision)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(stages).write(to: URL(fileURLWithPath: activationAudit))
            if verbose { stderr("Wrote activation audit to \(activationAudit)") }
            return
        }

        if verbose, inputs.count > 1 {
            let loadedModels = punctuate ? "ASR and punctuation models" : "ASR model"
            stderr("Loaded \(loadedModels) once; processing \(inputs.count) inputs.")
            if outputDirectory == nil { stderr("Multiple stdout results are emitted as JSON Lines.") }
        }

        for (index, input) in inputs.enumerated() {
            let totalStart = Date()
            let inputURL = URL(fileURLWithPath: input).standardizedFileURL
            let audioStart = Date()
            let audio = try GraniteAudioInput.load(url: inputURL)
            let audioLoadSeconds = Date().timeIntervalSince(audioStart)
            if verbose {
                stderr(String(format: "Loaded %@: %.2fs at %d Hz", inputURL.lastPathComponent, audio.duration, audio.sampleRate))
            }

            Memory.peakMemory = 0
            let inferenceStart = Date()
            let rawResult = try recognizer.transcribe(
                audio,
                activationPrecision: precision,
                ctcVocabularyTileSize: ctcVocabularyTile,
                middleCTCVocabularyTileSize: middleCTCVocabularyTile,
                audioChunkDuration: chunkDuration,
                audioChunkContext: chunkContext)
            let inferenceSeconds = Date().timeIntervalSince(inferenceStart)

            let punctuationStart = Date()
            let formatting = formatter?.format(rawResult.rawText)
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
            let report = BenchmarkResult(
                audioFile: inputURL.path,
                model: model,
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
                activationPrecision: precision.rawValue,
                ctcVocabularyTileSize: ctcVocabularyTile,
                middleCTCVocabularyTileSize: middleCTCVocabularyTile,
                audioChunkDurationSeconds: chunkDuration,
                audioChunkContextSeconds: chunkContext,
                mlxActiveMemoryBytes: memory.activeMemory,
                mlxCacheMemoryBytes: memory.cacheMemory,
                mlxPeakMemoryBytes: memory.peakMemory,
                mlxCacheLimitBytes: Memory.cacheLimit)
            let document = TranscriptionDocument(
                audioFile: inputURL.path,
                text: result.text,
                rawText: result.rawText,
                formattedText: result.formattedText,
                durationSeconds: result.duration,
                model: model,
                weightQuantizationBits: recognizer.artifact.configuration.quantization?.bits,
                punctuationModel: punctuate ? punctuationModel : nil,
                punctuationPrecision: formatter?.artifact.configuration.precision,
                punctuationQuantizationBits: formatter?.artifact.configuration.quantization?.bits,
                activationPrecision: precision.rawValue,
                timingNote: "Word timings are approximate CTC emission-frame alignments.",
                words: result.words,
                segments: segments,
                performance: report)

            try emit(
                format: format, document: document, transcription: result,
                segments: segments, inputURL: inputURL, inputIndex: index,
                outputDirectory: outputDirectory, resolver: resolver,
                forceJSONLine: outputDirectory == nil && inputs.count > 1)
            if verbose {
                stderr(String(format: "Inference %.3fs; formatting %.3fs; %.1fx realtime", inferenceSeconds, punctuationInferenceSeconds, report.realtimeMultiple))
            }
            if benchmark { stderr(try encodedJSON(report, pretty: false)) }
            Memory.clearCache()
        }
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
        forceJSONLine: Bool
    ) throws {
        if forceJSONLine {
            FileHandle.standardOutput.write(Data((try encodedJSON(document, pretty: false) + "\n").utf8))
            return
        }
        let formats: [OutputFormat] = format == .all ? [.txt, .srt, .vtt, .json] : [format]
        for selected in formats {
            let content = try rendered(selected, document: document, transcription: transcription, segments: segments)
            if let outputDirectory {
                let destination = resolver.destination(
                    directory: outputDirectory,
                    baseName: renderedBaseName(for: inputURL, index: inputIndex),
                    extension: selected.rawValue)
                try Data(content.utf8).write(to: destination, options: .atomic)
                if verbose { stderr("Wrote \(destination.path)") }
            } else {
                FileHandle.standardOutput.write(Data(content.utf8))
            }
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
