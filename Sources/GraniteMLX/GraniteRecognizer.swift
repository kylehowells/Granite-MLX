import Foundation
import MLX
import MLXNN

/// Affine weight-quantization settings for a Granite checkpoint.
public struct GraniteQuantizationConfiguration: Decodable, Sendable, Equatable {
    /// Default number of source values sharing a scale and bias.
    public var groupSize: Int
    /// Packed weight bit width.
    public var bits: Int
    /// MLX quantization mode.
    public var mode: QuantizationMode
    /// Optional per-module overrides. Keys use the flattened MLX module path,
    /// for example `encoder.input_linear`.
    public var groupSizes: [String: Int] = [:]

    private enum CodingKeys: String, CodingKey {
        case groupSize = "group_size", bits, mode
        case groupSizes = "group_sizes"
    }

    /// Creates an affine quantization configuration.
    public init(
        groupSize: Int, bits: Int, mode: QuantizationMode,
        groupSizes: [String: Int] = [:]
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.groupSizes = groupSizes
    }

    /// Decodes quantization settings from a converted checkpoint configuration.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        groupSize = try container.decode(Int.self, forKey: .groupSize)
        bits = try container.decode(Int.self, forKey: .bits)
        mode = try container.decode(QuantizationMode.self, forKey: .mode)
        groupSizes = try container.decodeIfPresent([String: Int].self, forKey: .groupSizes) ?? [:]
    }

    /// Returns the per-module group size override or the default group size.
    public func groupSize(for modulePath: String) -> Int {
        groupSizes[modulePath] ?? groupSize
    }
}

/// Runtime architecture configuration for Granite Speech 5.0 TurboCTC.
public struct GraniteModelConfiguration: Decodable, Sendable {
    /// Checkpoint architecture identifier.
    public var modelType: String = "granite_speech5_ctc"
    /// CTC tokenizer vocabulary size.
    public var vocabSize: Int = 16_384
    /// Encoder hidden width.
    public var hiddenSize: Int = 1_024
    /// Number of Conformer layers.
    public var numHiddenLayers: Int = 16
    /// Required input audio sample rate.
    public var sampleRate: Int = 16_000
    /// CTC output frames per second.
    public var outputFrameRate: Double = 12.5
    /// Feed-forward hidden width.
    public var intermediateSize: Int = 4_096
    /// Number of attention heads.
    public var numAttentionHeads: Int = 8
    /// Width of each attention head.
    public var headDimension: Int = 128
    /// Local attention context size.
    public var contextSize: Int = 128
    /// Convolution module expansion multiplier.
    public var convExpansionFactor: Int = 2
    /// Depthwise convolution kernel width.
    public var convKernelSize: Int = 7
    /// Encoder layers that perform temporal subsampling.
    public var subsampleLayers: [Int] = [0, 1]
    /// Relative-position embedding capacity.
    public var maxPositionEmbeddings: Int = 512
    /// Weight quantization settings, or `nil` for floating-point weights.
    public var quantization: GraniteQuantizationConfiguration?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type", vocabSize = "vocab_size", encoderConfig = "encoder_config"
        case quantization
    }

    private enum EncoderKeys: String, CodingKey {
        case hiddenSize = "hidden_size", numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size", numAttentionHeads = "num_attention_heads"
        case headDimension = "head_dim", contextSize = "context_size"
        case convExpansionFactor = "conv_expansion_factor", convKernelSize = "conv_kernel_size"
        case subsampleLayers = "subsample_layers"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    /// Creates the default Granite Speech 5.0 configuration.
    public init() {}

    /// Decodes architecture settings from a converted checkpoint configuration.
    public init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try root.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        vocabSize = try root.decodeIfPresent(Int.self, forKey: .vocabSize) ?? vocabSize
        quantization = try root.decodeIfPresent(
            GraniteQuantizationConfiguration.self, forKey: .quantization)
        if root.contains(.encoderConfig) {
            let encoder = try root.nestedContainer(keyedBy: EncoderKeys.self, forKey: .encoderConfig)
            hiddenSize = try encoder.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
            numHiddenLayers = try encoder.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? numHiddenLayers
            intermediateSize = try encoder.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
            numAttentionHeads = try encoder.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? numAttentionHeads
            headDimension = try encoder.decodeIfPresent(Int.self, forKey: .headDimension) ?? headDimension
            contextSize = try encoder.decodeIfPresent(Int.self, forKey: .contextSize) ?? contextSize
            convExpansionFactor = try encoder.decodeIfPresent(Int.self, forKey: .convExpansionFactor) ?? convExpansionFactor
            convKernelSize = try encoder.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? convKernelSize
            subsampleLayers = try encoder.decodeIfPresent([Int].self, forKey: .subsampleLayers) ?? subsampleLayers
            maxPositionEmbeddings = try encoder.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? maxPositionEmbeddings
        }
    }
}

/// A decoded word with approximate CTC timing.
public struct GraniteWord: Codable, Sendable, Equatable {
    /// Decoded word text.
    public let text: String
    /// Word onset in seconds.
    public let start: Double
    /// Word end in seconds.
    public let end: Double

    /// Creates a timed word.
    public init(text: String, start: Double, end: Double) {
        self.text = text
        self.start = start
        self.end = end
    }
}

/// Speech-recognition output containing raw, formatted, and timed forms.
public struct GraniteTranscription: Codable, Sendable {
    /// User-facing text. This is raw text until `applyingFormatting` is called.
    public let text: String
    /// Exact unformatted CTC text.
    public let rawText: String
    /// Punctuated text when formatting has been applied.
    public let formattedText: String?
    /// Timestamped CTC token emissions.
    public let tokens: [GraniteTokenTiming]
    /// Whitespace-grouped words with approximate timings.
    public let words: [GraniteWord]
    /// Input audio duration in seconds.
    public let duration: Double
    /// Model path or identifier used by the recognizer.
    public let model: String

    /// Creates a transcription result.
    public init(
        text: String,
        rawText: String? = nil,
        formattedText: String? = nil,
        tokens: [GraniteTokenTiming] = [],
        words: [GraniteWord],
        duration: Double,
        model: String
    ) {
        self.text = text
        self.rawText = rawText ?? text
        self.formattedText = formattedText
        self.tokens = tokens
        self.words = words
        self.duration = duration
        self.model = model
    }

    /// Returns a copy whose visible text and word spelling use formatter output.
    public func applyingFormatting(_ formatting: PunctuationFormattingResult) -> GraniteTranscription {
        let formattedWords = formatting.text.split(whereSeparator: \.isWhitespace).map(String.init)
        let timedWords: [GraniteWord]
        if formattedWords.count == words.count {
            timedWords = zip(words, formattedWords).map { word, formatted in
                GraniteWord(text: formatted, start: word.start, end: word.end)
            }
        } else {
            // Formatting is expected to preserve whitespace-delimited words.
            // Retain valid timing rather than inventing an unsafe alignment if
            // a future formatter checkpoint changes that contract.
            timedWords = words
        }
        return GraniteTranscription(
            text: formatting.text,
            rawText: rawText,
            formattedText: formatting.text,
            tokens: tokens,
            words: timedWords,
            duration: duration,
            model: model
        )
    }
}

/// Storage precision used between Granite encoder stages.
public enum GraniteActivationPrecision: String, Codable, Sendable, CaseIterable {
    /// Preserve the reference implementation's frontend dtype.
    case baseline
    /// Cast encoder input to FP16. Sensitive softmax operations remain FP32
    /// and are cast back by the model implementation.
    case fp16
    /// E4M3 byte storage between layers; computation remains FP16.
    case fp8Emulated = "fp8-emulated"
    /// Signed INT8 storage between layers; computation remains FP16.
    case int8Emulated = "int8-emulated"
}

/// A diagnostic snapshot of one evaluated activation tensor.
public struct GraniteActivationStage: Codable, Sendable {
    /// Runtime stage name.
    public let name: String
    /// Tensor dimensions.
    public let shape: [Int]
    /// MLX scalar type name.
    public let dtype: String
    /// Evaluated tensor byte count.
    public let evaluatedBytes: Int
    /// Minimum scalar value.
    public let minimum: Float
    /// Maximum scalar value.
    public let maximum: Float
}

/// Errors produced while loading or running the Granite recognizer.
public enum GraniteRecognizerError: Error, GraniteDiagnosticError {
    /// A checkpoint or runtime option uses an unsupported configuration.
    case unsupportedConfiguration(String)
    /// A checkpoint directory is missing required or valid model artifacts.
    case invalidModel(URL)

    /// Stable diagnostic identifier for the failure.
    public var diagnosticCode: String {
        switch self {
        case .unsupportedConfiguration: "GMLX-RUNTIME-001"
        case .invalidModel: "GMLX-RUNTIME-002"
        }
    }

    /// Low-level context useful for diagnostics.
    public var technicalDetails: String? {
        switch self {
        case .unsupportedConfiguration(let message): message
        case .invalidModel(let url): "model_directory=\(url.path)"
        }
    }

    /// User-facing localized failure description containing the diagnostic code.
    public var errorDescription: String? {
        switch self {
        case .unsupportedConfiguration:
            "[\(diagnosticCode)] The model or requested runtime configuration is not supported. Technical details: \(technicalDetails!)."
        case .invalidModel:
            "[\(diagnosticCode)] The model directory is incomplete, corrupt, or incompatible. Re-download the model or select another checkpoint. Technical details: \(technicalDetails!)."
        }
    }
}

/// Public runtime façade. The model graph is intentionally added behind this
/// stable API after the conversion manifest and tensor mapping are finalized.
public final class GraniteRecognizer: @unchecked Sendable {
    /// Materialized checkpoint directory.
    public let modelURL: URL
    /// Loaded checkpoint artifact and configuration.
    public let artifact: GraniteModelArtifact
    /// Frontend used to create Granite log-mel features.
    public let featureExtractor: GraniteFeatureExtractor
    private let model: GraniteCTCModel
    private let tokenizer: GraniteTokenizer

    /// Creates a recognizer from an already downloaded checkpoint directory.
    public init(modelURL: URL) throws {
        let artifact = try GraniteModelLoader.load(from: modelURL)
        let model = GraniteCTCModel(artifact.configuration)
        if let quantization = artifact.configuration.quantization {
            quantize(
                model: model,
                filter: { path, _ in
                    (
                        quantization.groupSize(for: path),
                        quantization.bits,
                        quantization.mode
                    )
                }
            )
        }
        model.update(parameters: ModuleParameters.unflattened(artifact.weights))
        model.train(false)
        MLX.eval(model.parameters())
        self.artifact = artifact
        self.modelURL = modelURL
        self.featureExtractor = GraniteFeatureExtractor(sampleRate: artifact.configuration.sampleRate)
        self.model = model
        do {
            self.tokenizer = try GraniteTokenizer(directory: modelURL)
        } catch let error as GraniteDiagnosticError {
            throw error
        } catch {
            throw GraniteOperationError.underlying(
                code: "GMLX-RUNTIME-005", operation: "Speech tokenizer loading",
                details: "directory=\(modelURL.path); error=\(String(reflecting: error))")
        }
    }

    /// Creates a recognizer from a local path, catalog alias, or Hugging Face ID.
    public convenience init(
        modelSource: String,
        hfToken: String? = nil,
        progressHandler: GraniteModelDownloadProgressHandler? = nil,
        cancellationToken: GraniteCancellationToken? = nil
    ) throws {
        let artifact = try GraniteModelLoader.load(
            source: modelSource, hfToken: hfToken,
            progressHandler: progressHandler, cancellationToken: cancellationToken)
        try self.init(modelURL: artifact.directory)
    }

    /// Exposes the model input tensor for frontend parity diagnostics.
    public func features(for audio: GraniteAudio) -> MLXArray {
        featureExtractor(audio.samples)
    }

    /// Transcribes prepared audio with optional chunk progress and cancellation.
    public func transcribe(
        _ audio: GraniteAudio,
        activationPrecision: GraniteActivationPrecision = .baseline,
        ctcVocabularyTileSize: Int = 0,
        middleCTCVocabularyTileSize: Int = 0,
        audioChunkDuration: Double = 0,
        audioChunkContext: Double = 0,
        cancellationToken: GraniteCancellationToken? = nil,
        progressHandler: GraniteOperationProgressHandler? = nil
    ) throws -> GraniteTranscription {
        try cancellationToken?.checkCancellation(operation: "Speech transcription")
        guard audio.sampleRate == artifact.configuration.sampleRate else {
            throw GraniteAudioError.invalidAudioFormat(
                details: "prepared_sample_rate=\(audio.sampleRate); required_sample_rate=\(artifact.configuration.sampleRate); source=\(audio.source.path)")
        }
        guard audio.samples.count > 256 else {
            throw GraniteAudioError.invalidAudioFormat(
                details: "prepared_sample_count=\(audio.samples.count); minimum_sample_count=257; source=\(audio.source.path)")
        }
        guard ctcVocabularyTileSize >= 0, middleCTCVocabularyTileSize >= 0,
              audioChunkDuration >= 0, audioChunkContext >= 0 else {
            throw GraniteRecognizerError.unsupportedConfiguration(
                "Tile sizes and chunk durations must be non-negative; ctc_tile=\(ctcVocabularyTileSize), middle_ctc_tile=\(middleCTCVocabularyTileSize), chunk_duration=\(audioChunkDuration), chunk_context=\(audioChunkContext).")
        }
        progressHandler?(GraniteOperationProgress(
            phase: .extractingFeatures, fractionCompleted: 0,
            message: "Preparing audio features"))
        if middleCTCVocabularyTileSize > 0,
           let quantization = artifact.configuration.quantization {
            let groupSize = quantization.groupSize(for: "encoder.out_mid")
            guard middleCTCVocabularyTileSize.isMultiple(of: groupSize) else {
                throw GraniteRecognizerError.unsupportedConfiguration(
                    "Middle CTC vocabulary tile size must be a multiple of \(groupSize) for this checkpoint."
                )
            }
        }
        let frameIDs: [Int]
        if audioChunkDuration > 0, audio.duration > audioChunkDuration {
            let chunkSampleCount = max(1, Int(audioChunkDuration * Double(audio.sampleRate)))
            let contextSampleCount = max(0, Int(audioChunkContext * Double(audio.sampleRate)))
            let outputFramesPerSecond = 12.5
            let coreStarts = Array(stride(
                from: 0, to: audio.samples.count, by: chunkSampleCount))
            var collectedFrameIDs: [Int] = []
            for (chunkIndex, coreStart) in coreStarts.enumerated() {
                    try cancellationToken?.checkCancellation(operation: "Speech transcription")
                    progressHandler?(GraniteOperationProgress(
                        phase: .transcribing,
                        fractionCompleted: Double(chunkIndex) / Double(coreStarts.count),
                        message: "Transcribing chunk \(chunkIndex + 1) of \(coreStarts.count)",
                        chunkIndex: chunkIndex, chunkCount: coreStarts.count))
                    let coreEnd = min(coreStart + chunkSampleCount, audio.samples.count)
                    let segmentStart = max(0, coreStart - contextSampleCount)
                    let segmentEnd = min(audio.samples.count, coreEnd + contextSampleCount)
                    let chunk = GraniteAudio(
                        samples: Array(audio.samples[segmentStart..<segmentEnd]),
                        sampleRate: audio.sampleRate,
                        source: audio.source
                    )
                    let segmentIDs = greedyFrameIDs(
                        for: chunk,
                        activationPrecision: activationPrecision,
                        ctcVocabularyTileSize: ctcVocabularyTileSize,
                        middleCTCVocabularyTileSize: middleCTCVocabularyTileSize
                    )
                    let leadingContextSeconds = Double(coreStart - segmentStart)
                        / Double(audio.sampleRate)
                    let coreSeconds = Double(coreEnd - coreStart) / Double(audio.sampleRate)
                    let firstFrame = min(
                        segmentIDs.count,
                        Int((leadingContextSeconds * outputFramesPerSecond).rounded())
                    )
                    let endFrame = min(
                        segmentIDs.count,
                        firstFrame + Int((coreSeconds * outputFramesPerSecond).rounded())
                    )
                    // Prevent differently-sized temporal buffers from
                    // accumulating across a long sequence of chunks.
                    Memory.clearCache()
                    collectedFrameIDs.append(contentsOf: segmentIDs[firstFrame..<endFrame])
            }
            frameIDs = collectedFrameIDs
        } else {
            progressHandler?(GraniteOperationProgress(
                phase: .transcribing, fractionCompleted: 0,
                message: "Transcribing audio", chunkIndex: 0, chunkCount: 1))
            frameIDs = greedyFrameIDs(
                for: audio,
                activationPrecision: activationPrecision,
                ctcVocabularyTileSize: ctcVocabularyTileSize,
                middleCTCVocabularyTileSize: middleCTCVocabularyTileSize
            )
        }
        try cancellationToken?.checkCancellation(operation: "Speech transcription")
        progressHandler?(GraniteOperationProgress(
            phase: .transcribing, fractionCompleted: 1,
            message: "Decoding transcript"))
        let tokenIDs = GraniteCTCDecoder.collapse(frameIDs)
        let tokenTimings = GraniteCTCDecoder.tokenTimings(
            frameIDs,
            frameRate: artifact.configuration.outputFrameRate,
            decodeToken: tokenizer.decodeToken)
        let transcription = GraniteTranscription(
            text: tokenizer.decode(tokenIDs),
            tokens: tokenTimings,
            words: GraniteCTCDecoder.words(from: tokenTimings),
            duration: audio.duration,
            model: modelURL.path
        )
        progressHandler?(GraniteOperationProgress(
            phase: .complete, fractionCompleted: 1,
            message: "Transcription complete"))
        return transcription
    }

    private func greedyFrameIDs(
        for audio: GraniteAudio,
        activationPrecision: GraniteActivationPrecision,
        ctcVocabularyTileSize: Int,
        middleCTCVocabularyTileSize: Int
    ) -> [Int] {
        let input = encoderInput(for: audio, activationPrecision: activationPrecision)
        return model.greedyFrameIDs(
            input,
            activationPrecision: activationPrecision,
            vocabularyTileSize: ctcVocabularyTileSize,
            middleVocabularyTileSize: middleCTCVocabularyTileSize
        ).squeezed(axis: 0).asArray(Int32.self).map(Int.init)
    }
    /// Evaluates and records key activation tensors for precision diagnostics.
    public func activationAudit(
        _ audio: GraniteAudio,
        activationPrecision: GraniteActivationPrecision = .baseline
    ) -> [GraniteActivationStage] {
        let input = encoderInput(for: audio, activationPrecision: activationPrecision)
        var stages: [GraniteActivationStage] = []
        func record(_ name: String, _ value: MLXArray) {
            let numeric = value.asType(.float32)
            let minimum = MLX.min(numeric)
            let maximum = MLX.max(numeric)
            MLX.eval(value, minimum, maximum)
            stages.append(GraniteActivationStage(
                name: name,
                shape: value.shape,
                dtype: String(describing: value.dtype),
                evaluatedBytes: value.size * value.dtype.size,
                minimum: minimum.item(Float.self),
                maximum: maximum.item(Float.self)
            ))
        }
        record("frontend", input)
        _ = model.forward(
            input, activationPrecision: activationPrecision, audit: record)
        return stages
    }

    private func encoderInput(
        for audio: GraniteAudio,
        activationPrecision: GraniteActivationPrecision
    ) -> MLXArray {
        let features = features(for: audio)
        switch activationPrecision {
        case .baseline:
            return features
        case .fp16, .fp8Emulated, .int8Emulated:
            return features.asType(.float16)
        }
    }
}
