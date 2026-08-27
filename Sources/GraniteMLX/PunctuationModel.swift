import Foundation
import Hub
import MLX
import Tokenizers

public struct PunctuationQuantizationConfiguration: Decodable, Sendable {
    public let bits: Int
    public let groupSize: Int
    public let mode: QuantizationMode

    private enum CodingKeys: String, CodingKey {
        case bits, mode
        case groupSize = "group_size"
    }
}

public struct PunctuationModelConfiguration: Decodable, Sendable {
    public let architecture: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let maxLength: Int
    public let layerNormEpsilon: Float
    public let precision: String
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

public struct PunctuationModelArtifact: @unchecked Sendable {
    public let directory: URL
    public let configuration: PunctuationModelConfiguration
    let weights: [String: MLXArray]
}

public enum PunctuationModelLoader {
    public static let defaultModelID = "iky1e/punctuation-fullstop-truecase-english-mlx-q8"

    public static func load(
        source: String = defaultModelID,
        hfToken: String? = nil,
        progressHandler: GraniteModelDownloadProgressHandler? = nil
    ) throws -> PunctuationModelArtifact {
        let localURL = URL(fileURLWithPath: source)
        if FileManager.default.fileExists(atPath: localURL.path) {
            return try load(from: localURL)
        }
        let directory = try GraniteModelCache.download(
            source, kind: .punctuation, hfToken: hfToken,
            progressHandler: progressHandler)
        return try load(from: directory)
    }

    public static func load(from directory: URL) throws -> PunctuationModelArtifact {
        let configURL = directory.appendingPathComponent("mlx_config.json")
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: configURL.path),
              FileManager.default.fileExists(atPath: weightsURL.path),
              FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw GraniteRecognizerError.invalidModel(directory)
        }
        let configuration = try JSONDecoder().decode(
            PunctuationModelConfiguration.self, from: Data(contentsOf: configURL))
        let weights = try MLX.loadArrays(url: weightsURL)
        let required = [
            "embeddings.word.weight", "embeddings.position.weight",
            "layers.0.query.weight", "layers.5.output.weight",
            "decoder.post.1.weight", "decoder.cap.1.weight",
        ]
        let missing = required.filter { weights[$0] == nil }
        guard missing.isEmpty else {
            throw GraniteRecognizerError.notImplemented(
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

public struct PunctuationFormattingResult: Sendable {
    public let text: String
    public let sentences: [String]
    /// Half-open whitespace-delimited word ranges for each sentence.
    public let sentenceWordRanges: [Range<Int>]

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

public final class PunctuationFormatter: @unchecked Sendable {
    public let artifact: PunctuationModelArtifact
    private let tokenizer: PunctuationTokenizer
    private let network: PunctuationNetwork

    public init(
        modelSource: String = PunctuationModelLoader.defaultModelID,
        hfToken: String? = nil,
        progressHandler: GraniteModelDownloadProgressHandler? = nil
    ) throws {
        let artifact = try PunctuationModelLoader.load(
            source: modelSource, hfToken: hfToken,
            progressHandler: progressHandler)
        self.artifact = artifact
        self.tokenizer = try PunctuationTokenizer(directory: artifact.directory)
        self.network = PunctuationNetwork(artifact: artifact)
    }

    public init(modelURL: URL) throws {
        let artifact = try PunctuationModelLoader.load(from: modelURL)
        self.artifact = artifact
        self.tokenizer = try PunctuationTokenizer(directory: artifact.directory)
        self.network = PunctuationNetwork(artifact: artifact)
    }

    public func format(_ text: String, overlap: Int = 16) -> PunctuationFormattingResult {
        let allIDs = tokenizer.encode(text.lowercased())
        guard !allIDs.isEmpty else {
            return PunctuationFormattingResult(text: "", sentences: [])
        }
        let payload = artifact.configuration.maxLength - 2
        let stride = payload - overlap
        var segments: [PunctuationPredictions] = []
        var start = 0
        while start < allIDs.count {
            let end = min(start + payload, allIDs.count)
            segments.append(network.predict([1] + Array(allIDs[start ..< end]) + [2]))
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
        return PunctuationFormattingResult(text: sentences.joined(separator: " "), sentences: sentences)
    }
}
