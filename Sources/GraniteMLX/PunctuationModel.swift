import Foundation
import Hub
import MLX
import Tokenizers

/// Quantization settings stored with an MLX punctuation checkpoint.
public struct PunctuationQuantizationConfiguration: Decodable, Sendable {
    /// Packed weight bit width.
    public let bits: Int
    /// Number of source values sharing each quantization scale.
    public let groupSize: Int
    /// MLX quantization mode.
    public let mode: QuantizationMode

    private enum CodingKeys: String, CodingKey {
        case bits, mode
        case groupSize = "group_size"
    }
}

/// Architecture metadata for the punctuation and true-casing model.
public struct PunctuationModelConfiguration: Decodable, Sendable {
    /// Architecture identifier written by the converter.
    public let architecture: String
    /// Transformer hidden width.
    public let hiddenSize: Int
    /// Feed-forward hidden width.
    public let intermediateSize: Int
    /// Number of transformer layers.
    public let numHiddenLayers: Int
    /// Number of attention heads.
    public let numAttentionHeads: Int
    /// Maximum tokenizer window size.
    public let maxLength: Int
    /// Layer-normalization epsilon.
    public let layerNormEpsilon: Float
    /// Source precision label.
    public let precision: String
    /// Quantization settings, or `nil` for floating-point weights.
    public let quantization: PunctuationQuantizationConfiguration?

    private enum CodingKeys: String, CodingKey {
        case architecture, precision, quantization
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case maxLength = "max_length"
        case layerNormEpsilon = "layer_norm_epsilon"
    }
}

/// A validated punctuation checkpoint loaded into MLX arrays.
public struct PunctuationModelArtifact: @unchecked Sendable {
    /// Local checkpoint directory.
    public let directory: URL
    /// Parsed architecture configuration.
    public let configuration: PunctuationModelConfiguration
    let weights: [String: MLXArray]
}

/// Loads punctuation checkpoints from disk or Hugging Face.
public enum PunctuationModelLoader {
    /// Recommended Q8 punctuation checkpoint.
    public static let defaultModelID = "iky1e/punctuation-fullstop-truecase-english-mlx-q8"

    /// Resolves, downloads if necessary, and loads a punctuation checkpoint.
    /// - Parameters:
    ///   - source: Local checkpoint directory, catalog alias, or Hugging Face
    ///     repository ID. The recommended Q8 formatter is used when omitted.
    ///   - storage: Explicit model and transfer-cache locations.
    ///   - hfToken: Optional Hugging Face token for private or gated repositories.
    ///   - progressHandler: Receives model cache and byte-weighted download events.
    ///   - cancellationToken: Cooperatively cancels model acquisition.
    /// - Returns: Validated formatter configuration and MLX tensors.
    /// - Throws: ``GraniteModelManagementError`` for download/cache failures,
    ///   ``GraniteRecognizerError`` for incompatible checkpoints, or
    ///   ``GraniteOperationError`` for cancellation and wrapped loading errors.
    public static func load(
        source: String = defaultModelID,
        storage: GraniteModelStorage = .default,
        hfToken: String? = nil,
        progressHandler: GraniteModelDownloadProgressHandler? = nil,
        cancellationToken: GraniteCancellationToken? = nil
    ) throws -> PunctuationModelArtifact {
        try cancellationToken?.checkCancellation(operation: "Punctuation model loading")
        let localURL = URL(fileURLWithPath: source)
        if FileManager.default.fileExists(atPath: localURL.path) {
            return try load(from: localURL)
        }
        let directory = try GraniteModelCache.download(
            source, kind: .punctuation, storage: storage, hfToken: hfToken,
            cancellationToken: cancellationToken, progressHandler: progressHandler)
        try cancellationToken?.checkCancellation(operation: "Punctuation model loading")
        return try load(from: directory)
    }

    /// Loads and validates a punctuation checkpoint directory.
    /// - Parameter directory: Directory containing `mlx_config.json`,
    ///   `model.safetensors`, and tokenizer files.
    /// - Returns: Parsed formatter configuration and loaded MLX tensors.
    /// - Throws: ``GraniteRecognizerError`` for missing files/tensors or
    ///   ``GraniteOperationError`` when JSON or safetensors loading fails.
    public static func load(from directory: URL) throws -> PunctuationModelArtifact {
        let configURL = directory.appendingPathComponent("mlx_config.json")
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: configURL.path),
              FileManager.default.fileExists(atPath: weightsURL.path),
              FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw GraniteRecognizerError.invalidModel(directory)
        }
        let configuration: PunctuationModelConfiguration
        let weights: [String: MLXArray]
        do {
            configuration = try JSONDecoder().decode(
                PunctuationModelConfiguration.self, from: Data(contentsOf: configURL))
            weights = try MLX.loadArrays(url: weightsURL)
        } catch let error as GraniteDiagnosticError {
            throw error
        } catch {
            throw GraniteOperationError.underlying(
                code: "GMLX-RUNTIME-004", operation: "Punctuation model loading",
                details: "directory=\(directory.path); error=\(error)")
        }
        let required = [
            "embeddings.word.weight", "embeddings.position.weight",
            "layers.0.query.weight", "layers.5.output.weight",
            "decoder.post.1.weight", "decoder.cap.1.weight",
        ]
        let missing = required.filter { weights[$0] == nil }
        guard missing.isEmpty else {
            throw GraniteRecognizerError.unsupportedConfiguration(
                "Punctuation checkpoint is missing tensors: \(missing.joined(separator: ", "))")
        }
        return PunctuationModelArtifact(
            directory: directory, configuration: configuration, weights: weights)
    }
}

private final class PunctuationTokenizer: @unchecked Sendable {
    private let tokenizer: any Tokenizer

    init(directory: URL) throws {
        let configData = try Data(
            contentsOf: directory.appendingPathComponent("tokenizer_config.json"))
        let tokenizerData = try Data(
            contentsOf: directory.appendingPathComponent("tokenizer.json"))
        self.tokenizer = try AutoTokenizer.from(
            tokenizerConfig: JSONDecoder().decode(Config.self, from: configData),
            tokenizerData: JSONDecoder().decode(Config.self, from: tokenizerData),
            strict: true)
    }

    func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    func piece(for id: Int) -> String {
        tokenizer.convertIdToToken(id) ?? ""
    }
}

private struct PunctuationPredictions {
    var ids: [Int]
    var pre: [Int]
    var post: [Int]
    var capitalizedCharacters: [[Bool]]
    var sentenceBoundary: [Bool]
}

private final class PunctuationNetwork: @unchecked Sendable {
    private let configuration: PunctuationModelConfiguration
    private let weights: [String: MLXArray]

    init(artifact: PunctuationModelArtifact) {
        configuration = artifact.configuration
        weights = artifact.weights
        MLX.eval(Array(weights.values))
    }

    private func linear(_ input: MLXArray, _ prefix: String) -> MLXArray {
        let weight = weights["\(prefix).weight"]!
        let output: MLXArray
        if let scales = weights["\(prefix).scales"],
           let quantization = configuration.quantization {
            output = quantizedMM(
                input, weight, scales: scales,
                biases: weights["\(prefix).biases"], transpose: true,
                groupSize: quantization.groupSize, bits: quantization.bits,
                mode: quantization.mode)
        } else {
            output = input.matmul(weight.transposed())
        }
        if let bias = weights["\(prefix).bias"] { return output + bias }
        return output
    }

    private func embedding(_ ids: MLXArray, _ prefix: String) -> MLXArray {
        let selectedWeight = take(weights["\(prefix).weight"]!, ids, axis: 0)
        guard let scales = weights["\(prefix).scales"],
              let quantization = configuration.quantization else {
            return selectedWeight
        }
        return dequantized(
            selectedWeight, scales: take(scales, ids, axis: 0),
            biases: weights["\(prefix).biases"].map { take($0, ids, axis: 0) },
            groupSize: quantization.groupSize, bits: quantization.bits,
            mode: quantization.mode)
    }

    private func norm(_ input: MLXArray, _ prefix: String) -> MLXArray {
        MLXFast.layerNorm(
            input, weight: weights["\(prefix).weight"],
            bias: weights["\(prefix).bias"], eps: configuration.layerNormEpsilon)
    }

    private func gelu(_ input: MLXArray) -> MLXArray {
        input * (1 + erf(input / Float(2).squareRoot())) / 2
    }

    func predict(_ tokenIDs: [Int]) -> PunctuationPredictions {
        let length = tokenIDs.count
        let ids = MLXArray(tokenIDs.map(Int32.init), [1, length])
        let positions = MLXArray.arange(length, dtype: .int32).expandedDimensions(axis: 0)
        let tokenTypes = MLXArray.zeros([1, length], dtype: .int32)
        var hidden = embedding(ids, "embeddings.word")
            + embedding(positions, "embeddings.position")
            + embedding(tokenTypes, "embeddings.token_type")
        hidden = norm(hidden, "embeddings.norm")

        let heads = configuration.numAttentionHeads
        let headSize = configuration.hiddenSize / heads
        for layer in 0 ..< configuration.numHiddenLayers {
            let prefix = "layers.\(layer)"
            let query = linear(hidden, "\(prefix).query")
                .reshaped([1, length, heads, headSize]).transposed(0, 2, 1, 3)
            let key = linear(hidden, "\(prefix).key")
                .reshaped([1, length, heads, headSize]).transposed(0, 2, 1, 3)
            let value = linear(hidden, "\(prefix).value")
                .reshaped([1, length, heads, headSize]).transposed(0, 2, 1, 3)
            let scores = query.matmul(key.transposed(0, 1, 3, 2)) / Float(headSize).squareRoot()
            let context = softmax(scores, axis: -1).matmul(value)
                .transposed(0, 2, 1, 3).reshaped([1, length, configuration.hiddenSize])
            hidden = norm(
                hidden + linear(context, "\(prefix).attention_output"),
                "\(prefix).attention_norm")
            let intermediate = gelu(linear(hidden, "\(prefix).intermediate"))
            hidden = norm(
                hidden + linear(intermediate, "\(prefix).output"),
                "\(prefix).output_norm")
        }

        let postLogits = linear(
            MLX.maximum(linear(hidden, "decoder.post.0"), MLXArray(0)),
            "decoder.post.1")
        let preLogits = linear(
            MLX.maximum(linear(hidden, "decoder.pre.0"), MLXArray(0)),
            "decoder.pre.1")
        let post = postLogits.argMax(axis: -1)
        let punctuationEmbedding = take(
            weights["decoder.punctuation_embedding.weight"]!, post, axis: 0)
        let segmentationInput = MLX.concatenated([hidden, punctuationEmbedding], axis: -1)
        let segmentationLogits = linear(
            MLX.maximum(linear(segmentationInput, "decoder.seg.0"), MLXArray(0)),
            "decoder.seg.1")
        let segmentation = segmentationLogits.argMax(axis: -1)
        let shiftedSegmentation = MLX.concatenated(
            [MLXArray.zeros([1, 1], dtype: segmentation.dtype), segmentation[0..., 0 ..< (length - 1)]],
            axis: 1).expandedDimensions(axis: -1).asType(hidden.dtype)
        let capitalizationInput = MLX.concatenated([hidden, shiftedSegmentation], axis: -1)
        let capitalization = linear(
            MLX.maximum(linear(capitalizationInput, "decoder.cap.0"), MLXArray(0)),
            "decoder.cap.1") .> 0

        MLX.eval(preLogits, post, segmentation, capitalization)
        let preValues = preLogits.argMax(axis: -1).asArray(Int32.self).map(Int.init)
        let postValues = post.asArray(Int32.self).map(Int.init)
        let segmentationValues = segmentation.asArray(Int32.self).map { $0 != 0 }
        let capValues = capitalization.asArray(Bool.self)
        let capRows = (0 ..< length).map { index in
            Array(capValues[(index * 16) ..< ((index + 1) * 16)])
        }
        return PunctuationPredictions(
            ids: tokenIDs, pre: preValues, post: postValues,
            capitalizedCharacters: capRows, sentenceBoundary: segmentationValues)
    }
}

/// Formatted transcript text and its sentence-to-word mapping.
public struct PunctuationFormattingResult: Sendable {
    /// Fully formatted text.
    public let text: String
    /// Individual formatted sentences.
    public let sentences: [String]
    /// Half-open whitespace-delimited word ranges for each sentence.
    public let sentenceWordRanges: [Range<Int>]

    /// Creates a result, deriving contiguous word ranges when omitted.
    /// - Parameters:
    ///   - text: Complete formatted transcript.
    ///   - sentences: Formatted sentences in display order.
    ///   - sentenceWordRanges: Half-open ranges into whitespace-delimited words.
    ///     When omitted, contiguous ranges are derived from `sentences`.
    public init(text: String, sentences: [String], sentenceWordRanges: [Range<Int>]? = nil) {
        self.text = text
        self.sentences = sentences
        if let sentenceWordRanges {
            self.sentenceWordRanges = sentenceWordRanges
        } else {
            var offset = 0
            self.sentenceWordRanges = sentences.map { sentence in
                let count = sentence.split(whereSeparator: \.isWhitespace).count
                defer { offset += count }
                return offset ..< (offset + count)
            }
        }
    }
}

/// Restores punctuation, capitalization, and sentence boundaries with MLX.
public final class PunctuationFormatter: GraniteTranscriptFormatter, @unchecked Sendable {
    /// Loaded formatter checkpoint.
    public let artifact: PunctuationModelArtifact
    private let tokenizer: PunctuationTokenizer
    private let network: PunctuationNetwork

    /// Architecture-independent metadata exposed to formatter clients.
    public var formatterInfo: GraniteTranscriptFormatterInfo {
        GraniteTranscriptFormatterInfo(
            architecture: artifact.configuration.architecture,
            precision: artifact.configuration.precision,
            quantizationBits: artifact.configuration.quantization?.bits)
    }

    /// Creates a formatter from a local path or Hugging Face repository ID.
    /// - Parameters:
    ///   - modelSource: Local checkpoint directory, catalog alias, or Hugging
    ///     Face repository ID. The recommended Q8 formatter is used when omitted.
    ///   - storage: Explicit model and transfer-cache locations.
    ///   - hfToken: Optional Hugging Face token for private or gated repositories.
    ///   - progressHandler: Receives model cache and download progress.
    ///   - cancellationToken: Cooperatively cancels model acquisition.
    /// - Throws: Model-management, checkpoint-validation, tokenizer-loading, or
    ///   cancellation errors produced while constructing the formatter.
    public init(
        modelSource: String = PunctuationModelLoader.defaultModelID,
        storage: GraniteModelStorage = .default,
        hfToken: String? = nil,
        progressHandler: GraniteModelDownloadProgressHandler? = nil,
        cancellationToken: GraniteCancellationToken? = nil
    ) throws {
        let artifact = try PunctuationModelLoader.load(
            source: modelSource, storage: storage, hfToken: hfToken,
            progressHandler: progressHandler, cancellationToken: cancellationToken)
        self.artifact = artifact
        do {
            self.tokenizer = try PunctuationTokenizer(directory: artifact.directory)
        } catch let error as GraniteDiagnosticError {
            throw error
        } catch {
            throw GraniteOperationError.underlying(
                code: "GMLX-RUNTIME-006", operation: "Punctuation tokenizer loading",
                details: "directory=\(artifact.directory.path); error=\(String(reflecting: error))")
        }
        self.network = PunctuationNetwork(artifact: artifact)
    }

    /// Creates a formatter from an already downloaded checkpoint directory.
    /// - Parameter modelURL: Materialized formatter repository directory.
    /// - Throws: ``GraniteRecognizerError`` for invalid checkpoint contents or
    ///   ``GraniteOperationError`` when model/tokenizer loading fails.
    public init(modelURL: URL) throws {
        let artifact = try PunctuationModelLoader.load(from: modelURL)
        self.artifact = artifact
        do {
            self.tokenizer = try PunctuationTokenizer(directory: artifact.directory)
        } catch let error as GraniteDiagnosticError {
            throw error
        } catch {
            throw GraniteOperationError.underlying(
                code: "GMLX-RUNTIME-006", operation: "Punctuation tokenizer loading",
                details: "directory=\(artifact.directory.path); error=\(String(reflecting: error))")
        }
        self.network = PunctuationNetwork(artifact: artifact)
    }

    /// Formats text without cancellation or progress reporting.
    /// - Parameters:
    ///   - text: Raw lowercased recognition text to annotate.
    ///   - overlap: Number of tokenizer items shared by adjacent windows. Must
    ///     be nonnegative and smaller than the model payload length.
    /// - Returns: Formatted text, sentences, and sentence word ranges.
    public func format(_ text: String, overlap: Int = 16) -> PunctuationFormattingResult {
        // The cancellable implementation can only throw when a supplied token
        // is cancelled, so this compatibility entry point is non-throwing.
        try! format(
            text, overlap: overlap, cancellationToken: nil,
            progressHandler: nil)
    }

    /// Formats text through the architecture-independent formatter protocol.
    /// - Parameters:
    ///   - text: Raw lexical transcript to annotate non-destructively.
    ///   - cancellationToken: Optional cooperative cancellation token.
    ///   - progressHandler: Receives formatter-window and completion progress.
    /// - Returns: Formatted text, sentences, and sentence word ranges.
    /// - Throws: ``GraniteOperationError`` when cancelled or an underlying
    ///   operation fails, and ``GraniteRecognizerError`` for invalid options.
    public func format(
        _ text: String,
        cancellationToken: GraniteCancellationToken?,
        progressHandler: GraniteOperationProgressHandler?
    ) throws -> PunctuationFormattingResult {
        try format(
            text, overlap: 16,
            cancellationToken: cancellationToken,
            progressHandler: progressHandler)
    }

    /// Restores punctuation, capitalization, and sentence boundaries.
    ///
    /// - Parameters:
    ///   - text: Raw lowercased speech-recognition text.
    ///   - overlap: Token overlap between formatter windows.
    ///   - cancellationToken: Optional cooperative cancellation token.
    ///   - progressHandler: Optional application-facing progress callback.
    /// - Returns: Formatted text and sentence word ranges.
    /// - Throws: ``GraniteRecognizerError/unsupportedConfiguration(_:)`` when
    ///   `overlap` is outside the supported range, or
    ///   ``GraniteOperationError/cancelled(operation:)`` when cancelled.
    public func format(
        _ text: String,
        overlap: Int = 16,
        cancellationToken: GraniteCancellationToken?,
        progressHandler: GraniteOperationProgressHandler?
    ) throws -> PunctuationFormattingResult {
        try cancellationToken?.checkCancellation(operation: "Text formatting")
        progressHandler?(GraniteOperationProgress(
            phase: .formatting, fractionCompleted: 0,
            message: "Restoring punctuation and capitalization"))
        let allIDs = tokenizer.encode(text.lowercased())
        guard !allIDs.isEmpty else {
            progressHandler?(GraniteOperationProgress(
                phase: .complete, fractionCompleted: 1,
                message: "Formatting complete"))
            return PunctuationFormattingResult(text: "", sentences: [])
        }
        let payload = artifact.configuration.maxLength - 2
        guard overlap >= 0, overlap < payload else {
            throw GraniteRecognizerError.unsupportedConfiguration(
                "Punctuation overlap must be in 0..<\(payload); received \(overlap).")
        }
        let stride = payload - overlap
        var segments: [PunctuationPredictions] = []
        var start = 0
        while start < allIDs.count {
            try cancellationToken?.checkCancellation(operation: "Text formatting")
            let end = min(start + payload, allIDs.count)
            segments.append(network.predict([1] + Array(allIDs[start ..< end]) + [2]))
            progressHandler?(GraniteOperationProgress(
                phase: .formatting,
                fractionCompleted: Double(end) / Double(allIDs.count),
                message: "Formatting token window \(segments.count)"))
            if end == allIDs.count { break }
            start += stride
        }

        var merged = PunctuationPredictions(
            ids: [], pre: [], post: [], capitalizedCharacters: [], sentenceBoundary: [])
        for (index, segment) in segments.enumerated() {
            let payloadCount = segment.ids.count - 2
            let lower = 1 + (index > 0 ? overlap / 2 : 0)
            let upper = 1 + payloadCount - (index < segments.count - 1 ? overlap / 2 : 0)
            merged.ids.append(contentsOf: segment.ids[lower ..< upper])
            merged.pre.append(contentsOf: segment.pre[lower ..< upper])
            merged.post.append(contentsOf: segment.post[lower ..< upper])
            merged.capitalizedCharacters.append(contentsOf: segment.capitalizedCharacters[lower ..< upper])
            merged.sentenceBoundary.append(contentsOf: segment.sentenceBoundary[lower ..< upper])
        }

        var sentences: [String] = []
        var current = ""
        for tokenIndex in merged.ids.indices {
            let piece = tokenizer.piece(for: merged.ids[tokenIndex])
            let characters = Array(piece)
            let beginsWord = characters.first == "▁"
            if beginsWord && !current.isEmpty { current.append(" ") }
            let characterStart = beginsWord ? 1 : 0
            guard characterStart < characters.count else { continue }
            for characterIndex in characterStart ..< characters.count {
                if characterIndex == characterStart && merged.pre[tokenIndex] == 1 {
                    current.append("¿")
                }
                var character = String(characters[characterIndex])
                if characterIndex < 16,
                   merged.capitalizedCharacters[tokenIndex][characterIndex] {
                    character = character.uppercased()
                }
                current.append(character)
                let post = merged.post[tokenIndex]
                if post == 1 {
                    current.append(".")
                } else if characterIndex == characters.count - 1 {
                    switch post {
                    case 2: current.append(".")
                    case 3: current.append(",")
                    case 4: current.append("?")
                    default: break
                    }
                }
            }
            if merged.sentenceBoundary[tokenIndex] && !current.isEmpty {
                sentences.append(current)
                current = ""
            }
        }
        if !current.isEmpty { sentences.append(current) }
        try cancellationToken?.checkCancellation(operation: "Text formatting")
        let result = Self.preservingOriginalWords(
            originalText: text, formattedSentences: sentences)
        progressHandler?(GraniteOperationProgress(
            phase: .complete, fractionCompleted: 1,
            message: "Formatting complete"))
        return result
    }

    static func preservingOriginalWords(
        originalText: String,
        formattedSentences: [String]
    ) -> PunctuationFormattingResult {
        let originalWords = originalText.split(whereSeparator: \.isWhitespace).map(String.init)
        let sentenceWords = formattedSentences.map {
            $0.split(whereSeparator: \.isWhitespace).map(String.init)
        }
        let formattedWords = sentenceWords.flatMap { $0 }

        // A formatter is an annotation stage, not a text-generation stage. If
        // its output cannot be aligned one-to-one, preserve all ASR text rather
        // than silently deleting or inventing words.
        guard originalWords.count == formattedWords.count else {
            return PunctuationFormattingResult(
                text: originalWords.joined(separator: " "),
                sentences: originalWords.isEmpty ? [] : [originalWords.joined(separator: " ")])
        }

        let preservedWords = zip(originalWords, formattedWords).map {
            preserveOriginalWord($0.0, formattedWord: $0.1)
        }
        var offset = 0
        let preservedSentences = sentenceWords.map { words -> String in
            defer { offset += words.count }
            return preservedWords[offset ..< (offset + words.count)].joined(separator: " ")
        }
        return PunctuationFormattingResult(
            text: preservedSentences.joined(separator: " "),
            sentences: preservedSentences)
    }

    private static func preserveOriginalWord(
        _ originalWord: String,
        formattedWord: String
    ) -> String {
        let unknownMarker = "<unk>"
        let containsUnknown = formattedWord.range(
            of: unknownMarker, options: .caseInsensitive) != nil
        let withoutUnknown = formattedWord.replacingOccurrences(
            of: unknownMarker, with: "", options: .caseInsensitive)
        let originalLexical = lexicalCharacters(in: originalWord)
        let formattedLexical = lexicalCharacters(in: withoutUnknown)

        // Genuine lexical changes are unsafe. Do not borrow punctuation from a
        // word that cannot be aligned to the recognizer's original characters.
        guard originalLexical.map({ String($0).lowercased() })
            == formattedLexical.map({ String($0).lowercased() }) else {
            return originalWord
        }
        guard containsUnknown else { return formattedWord }

        var caseIndex = 0
        var repaired = ""
        for character in originalWord {
            if character.isLetter || character.isNumber {
                let formattedCharacter = formattedLexical[caseIndex]
                repaired.append(formattedCharacter.isUppercase
                    ? String(character).uppercased()
                    : String(character).lowercased())
                caseIndex += 1
            } else {
                repaired.append(character)
            }
        }

        if withoutUnknown.hasPrefix("¿"), !repaired.hasPrefix("¿") {
            repaired = "¿" + repaired
        }
        if let trailing = withoutUnknown.last,
           [".", ",", "?"].contains(trailing),
           repaired.last != trailing {
            repaired.append(trailing)
        }
        return repaired
    }

    private static func lexicalCharacters(in word: String) -> [Character] {
        word.filter { $0.isLetter || $0.isNumber }
    }
}
