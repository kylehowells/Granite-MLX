import Foundation
import MLX
import MLXNN

public struct GraniteQuantizationConfiguration: Decodable, Sendable, Equatable {
    public var groupSize: Int
    public var bits: Int
    public var mode: QuantizationMode
    /// Optional per-module overrides. Keys use the flattened MLX module path,
    /// for example `encoder.input_linear`.
    public var groupSizes: [String: Int] = [:]

    private enum CodingKeys: String, CodingKey {
        case groupSize = "group_size", bits, mode
        case groupSizes = "group_sizes"
    }

    public init(
        groupSize: Int, bits: Int, mode: QuantizationMode,
        groupSizes: [String: Int] = [:]
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.groupSizes = groupSizes
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        groupSize = try container.decode(Int.self, forKey: .groupSize)
        bits = try container.decode(Int.self, forKey: .bits)
        mode = try container.decode(QuantizationMode.self, forKey: .mode)
        groupSizes = try container.decodeIfPresent([String: Int].self, forKey: .groupSizes) ?? [:]
    }

    public func groupSize(for modulePath: String) -> Int {
        groupSizes[modulePath] ?? groupSize
    }
}

public struct GraniteModelConfiguration: Decodable, Sendable {
    public var modelType: String = "granite_speech5_ctc"
    public var vocabSize: Int = 16_384
    public var hiddenSize: Int = 1_024
    public var numHiddenLayers: Int = 16
    public var sampleRate: Int = 16_000
    public var outputFrameRate: Double = 12.5
    public var intermediateSize: Int = 4_096
    public var numAttentionHeads: Int = 8
    public var headDimension: Int = 128
    public var contextSize: Int = 128
    public var convExpansionFactor: Int = 2
    public var convKernelSize: Int = 7
    public var subsampleLayers: [Int] = [0, 1]
    public var maxPositionEmbeddings: Int = 512
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

    public init() {}

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

public struct GraniteWord: Codable, Sendable, Equatable {
    public let text: String
    public let start: Double
    public let end: Double

    public init(text: String, start: Double, end: Double) {
        self.text = text
        self.start = start
        self.end = end
    }
}

public struct GraniteTranscription: Codable, Sendable {
    /// User-facing text. This is raw text until `applyingFormatting` is called.
    public let text: String
    public let rawText: String
    public let formattedText: String?
    public let tokens: [GraniteTokenTiming]
    public let words: [GraniteWord]
    public let duration: Double
    public let model: String

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

public struct GraniteActivationStage: Codable, Sendable {
    public let name: String
    public let shape: [Int]
    public let dtype: String
    public let evaluatedBytes: Int
    public let minimum: Float
    public let maximum: Float
}

public enum GraniteRecognizerError: Error, LocalizedError {
    case notImplemented(String)
    case invalidModel(URL)

    public var errorDescription: String? {
        switch self {
        case .notImplemented(let message): message
        case .invalidModel(let url): "Invalid Granite model directory: \(url.path)"
        }
    }
}

/// Public runtime façade. The model graph is intentionally added behind this
/// stable API after the conversion manifest and tensor mapping are finalized.
public final class GraniteRecognizer: @unchecked Sendable {
    public let modelURL: URL
    public let artifact: GraniteModelArtifact
    public let featureExtractor: GraniteFeatureExtractor
    private let model: GraniteCTCModel
    private let tokenizer: GraniteTokenizer

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
        self.tokenizer = try GraniteTokenizer(directory: modelURL)
    }

    public convenience init(
        modelSource: String,
        hfToken: String? = nil,
        progressHandler: GraniteModelDownloadProgressHandler? = nil
    ) throws {
        let artifact = try GraniteModelLoader.load(
            source: modelSource, hfToken: hfToken,
            progressHandler: progressHandler)
        try self.init(modelURL: artifact.directory)
    }

    /// Exposes the model input tensor for frontend parity diagnostics.
    public func features(for audio: GraniteAudio) -> MLXArray {
        featureExtractor(audio.samples)
    }

    public func transcribe(
        _ audio: GraniteAudio,
        activationPrecision: GraniteActivationPrecision = .baseline,
        ctcVocabularyTileSize: Int = 0,
        middleCTCVocabularyTileSize: Int = 0,
        audioChunkDuration: Double = 0,
        audioChunkContext: Double = 0
    ) throws -> GraniteTranscription {
        if middleCTCVocabularyTileSize > 0,
           let quantization = artifact.configuration.quantization {
            let groupSize = quantization.groupSize(for: "encoder.out_mid")
            guard middleCTCVocabularyTileSize.isMultiple(of: groupSize) else {
                throw GraniteRecognizerError.notImplemented(
                    "Middle CTC vocabulary tile size must be a multiple of \(groupSize) for this checkpoint."
                )
            }
        }
        let frameIDs: [Int]
        if audioChunkDuration > 0, audio.duration > audioChunkDuration {
            let chunkSampleCount = max(1, Int(audioChunkDuration * Double(audio.sampleRate)))
            let contextSampleCount = max(0, Int(audioChunkContext * Double(audio.sampleRate)))
            let outputFramesPerSecond = 12.5
            frameIDs = stride(from: 0, to: audio.samples.count, by: chunkSampleCount)
                .flatMap { coreStart -> [Int] in
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
                    return Array(segmentIDs[firstFrame..<endFrame])
                }
        } else {
            frameIDs = greedyFrameIDs(
                for: audio,
                activationPrecision: activationPrecision,
                ctcVocabularyTileSize: ctcVocabularyTileSize,
                middleCTCVocabularyTileSize: middleCTCVocabularyTileSize
            )
        }
        let tokenIDs = GraniteCTCDecoder.collapse(frameIDs)
        let tokenTimings = GraniteCTCDecoder.tokenTimings(
            frameIDs,
            frameRate: artifact.configuration.outputFrameRate,
            decodeToken: tokenizer.decodeToken)
        return GraniteTranscription(
            text: tokenizer.decode(tokenIDs),
            tokens: tokenTimings,
            words: GraniteCTCDecoder.words(from: tokenTimings),
            duration: audio.duration,
            model: modelURL.path
        )
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
