import Foundation
import MLX
import MLXNN

final class GraniteEvalBatchNorm: Module, @unchecked Sendable {
    var weight: MLXArray
    var bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray
    let eps: Float

    init(featureCount: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([featureCount])
        self.bias = MLXArray.zeros([featureCount])
        self._runningMean.wrappedValue = MLXArray.zeros([featureCount])
        self._runningVar.wrappedValue = MLXArray.ones([featureCount])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x - runningMean) / MLX.sqrt(runningVar + eps) * weight + bias
    }
}

final class GraniteFeedForward: Module, @unchecked Sendable {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(_ config: GraniteModelConfiguration) {
        self._linear1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        self._linear2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

final class GraniteBlockAttention: Module, @unchecked Sendable {
    let numHeads: Int
    let headDimension: Int
    let contextSize: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "rel_pos_emb") var relativePositionEmbedding: Embedding

    init(_ config: GraniteModelConfiguration) {
        let inner = config.numAttentionHeads * config.headDimension
        self.numHeads = config.numAttentionHeads
        self.headDimension = config.headDimension
        self.contextSize = config.contextSize
        self.scale = pow(Float(config.headDimension), -0.5)
        self._qProj.wrappedValue = Linear(config.hiddenSize, inner, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, inner, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, inner, bias: false)
        self._oProj.wrappedValue = Linear(inner, config.hiddenSize)
        self._relativePositionEmbedding.wrappedValue = Embedding(
            embeddingCount: 2 * config.maxPositionEmbeddings + 1,
            dimensions: config.headDimension
        )
    }

    func callAsFunction(_ x: MLXArray, attentionDistances: MLXArray) -> MLXArray {
        let length = x.dim(1)
        let fullLength = (length / contextSize) * contextSize
        let q = qProj(x)
        let k = kProj(x)
        let v = vProj(x)
        var outputs: [MLXArray] = []
        if fullLength > 0 {
            outputs.append(attend(
                q: q[0..., 0..<fullLength, 0...],
                k: k[0..., 0..<fullLength, 0...],
                v: v[0..., 0..<fullLength, 0...],
                blockLength: contextSize,
                attentionDistances: attentionDistances
            ))
        }
        if fullLength < length {
            let tail = length - fullLength
            outputs.append(attend(
                q: q[0..., fullLength..<length, 0...],
                k: k[0..., fullLength..<length, 0...],
                v: v[0..., fullLength..<length, 0...],
                blockLength: tail,
                attentionDistances: attentionDistances
            ))
        }
        return oProj(outputs.count == 1 ? outputs[0] : MLX.concatenated(outputs, axis: 1))
    }

    private func attend(
        q: MLXArray, k: MLXArray, v: MLXArray,
        blockLength: Int, attentionDistances: MLXArray
    ) -> MLXArray {
        let batch = q.dim(0)
        let blockCount = q.dim(1) / blockLength
        func shaped(_ value: MLXArray) -> MLXArray {
            value.reshaped(batch, blockCount, blockLength, numHeads, headDimension)
                .transposed(0, 1, 3, 2, 4)
        }
        let query = shaped(q)
        let key = shaped(k)
        let value = shaped(v)
        let distances = attentionDistances[0..<blockLength, 0..<blockLength]
        let relative = relativePositionEmbedding(distances)
        // Avoid materializing [B, blocks, heads, query, key, headDim]. This
        // contraction is the key to keeping hour-long inputs practical.
        let relativeBias = MLX.einsum("bmhcd,crd->bmhcr", query, relative) * scale
        var weights = query.matmul(key.transposed(0, 1, 2, 4, 3)) * scale + relativeBias
        weights = softmax(weights.asType(.float32), axis: -1).asType(query.dtype)
        return weights.matmul(value)
            .transposed(0, 1, 3, 2, 4)
            .reshaped(batch, blockCount * blockLength, numHeads * headDimension)
    }
}

final class GraniteConvolution: Module, @unchecked Sendable {
    let stride: Int
    let kernelSize: Int
    @ModuleInfo(key: "pointwise_lin1") var pointwiseLinear1: Linear
    @ModuleInfo(key: "depthwise_conv") var depthwiseConvolution: Conv1d
    @ModuleInfo(key: "norm") var normalization: GraniteEvalBatchNorm
    @ModuleInfo(key: "pointwise_lin2") var pointwiseLinear2: Linear

    init(_ config: GraniteModelConfiguration, stride: Int) {
        let inner = config.hiddenSize * config.convExpansionFactor
        self.stride = stride
        self.kernelSize = config.convKernelSize
        self._pointwiseLinear1.wrappedValue = Linear(config.hiddenSize, inner * 2)
        self._depthwiseConvolution.wrappedValue = Conv1d(
            inputChannels: inner, outputChannels: inner,
            kernelSize: config.convKernelSize, stride: stride,
            padding: 0, groups: inner, bias: false
        )
        self._normalization.wrappedValue = GraniteEvalBatchNorm(featureCount: inner)
        self._pointwiseLinear2.wrappedValue = Linear(inner, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = pointwiseLinear1(x)
        let parts = projected.split(parts: 2, axis: -1)
        var hidden = parts[0] * sigmoid(parts[1])
        let left = kernelSize / 2
        let right = left - (kernelSize + 1) % 2
        hidden = MLX.padded(hidden, widths: [.init((0, 0)), .init((left, right)), .init((0, 0))])
        hidden = depthwiseConvolution(hidden)
        hidden = silu(normalization(hidden))
        return pointwiseLinear2(hidden)
    }
}

final class GraniteConformerLayer: Module, @unchecked Sendable {
    let subsamples: Bool
    @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var feedForward1: GraniteFeedForward
    @ModuleInfo(key: "norm_self_att") var normSelfAttention: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttention: GraniteBlockAttention
    @ModuleInfo(key: "norm_conv") var normConvolution: LayerNorm
    @ModuleInfo(key: "conv") var convolution: GraniteConvolution
    @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var feedForward2: GraniteFeedForward
    @ModuleInfo(key: "norm_out") var normOutput: LayerNorm

    init(_ config: GraniteModelConfiguration, subsamples: Bool) {
        self.subsamples = subsamples
        self._normFeedForward1.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        self._feedForward1.wrappedValue = GraniteFeedForward(config)
        self._normSelfAttention.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        self._selfAttention.wrappedValue = GraniteBlockAttention(config)
        self._normConvolution.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        self._convolution.wrappedValue = GraniteConvolution(config, stride: subsamples ? 2 : 1)
        self._normFeedForward2.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        self._feedForward2.wrappedValue = GraniteFeedForward(config)
        self._normOutput.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
    }

    func callAsFunction(_ input: MLXArray, attentionDistances: MLXArray) -> MLXArray {
        var hidden = input + 0.5 * feedForward1(normFeedForward1(input))
        hidden = hidden + selfAttention(normSelfAttention(hidden), attentionDistances: attentionDistances)
        let convolutionOutput = convolution(normConvolution(hidden))
        if subsamples {
            let half = hidden.dim(1) / 2
            hidden = hidden[0..., 0..<(half * 2), 0...]
                .reshaped(hidden.dim(0), half, 2, hidden.dim(2))
                .mean(axis: 2)
            hidden = hidden + convolutionOutput[0..., 0..<half, 0...]
        } else {
            hidden = hidden + convolutionOutput
        }
        hidden = hidden + 0.5 * feedForward2(normFeedForward2(hidden))
        return normOutput(hidden)
    }
}

final class GraniteEncoder: Module, @unchecked Sendable {
    @ModuleInfo(key: "input_linear") var inputLinear: Linear
    @ModuleInfo(key: "layers") var layers: [GraniteConformerLayer]
    @ModuleInfo(key: "out") var output: Linear
    @ModuleInfo(key: "out_mid") var middleOutput: Linear
    let attentionDistances: MLXArray

    init(_ config: GraniteModelConfiguration) {
        self._inputLinear.wrappedValue = Linear(320, config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map {
            GraniteConformerLayer(config, subsamples: config.subsampleLayers.contains($0))
        }
        self._output.wrappedValue = Linear(config.hiddenSize, config.vocabSize)
        self._middleOutput.wrappedValue = Linear(config.vocabSize, config.hiddenSize)
        let positions = MLXArray(Int32(0)..<Int32(config.contextSize))
        let distances = positions.expandedDimensions(axis: 1) - positions.expandedDimensions(axis: 0)
        self.attentionDistances = MLX.clip(
            distances, min: -config.contextSize, max: config.contextSize
        ) + config.maxPositionEmbeddings
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        forward(features)
    }

    func forward(
        _ features: MLXArray,
        activationPrecision: GraniteActivationPrecision = .baseline,
        audit: ((String, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        let hidden = encode(
            features, activationPrecision: activationPrecision, audit: audit)
        let logits = output(hidden)
        audit?("encoder.final_logits", logits)
        return logits
    }

    func greedyFrameIDs(
        _ features: MLXArray,
        activationPrecision: GraniteActivationPrecision = .baseline,
        vocabularyTileSize: Int,
        middleVocabularyTileSize: Int = 0
    ) -> MLXArray {
        let hidden = encode(
            features,
            activationPrecision: activationPrecision,
            middleVocabularyTileSize: middleVocabularyTileSize,
            audit: nil)
        guard vocabularyTileSize > 0, vocabularyTileSize < output.shape.0 else {
            return output(hidden).argMax(axis: -1)
        }

        var bestValues: MLXArray?
        var bestIDs: MLXArray?
        let vocabularySize = output.shape.0
        for start in stride(from: 0, to: vocabularySize, by: vocabularyTileSize) {
            let end = min(start + vocabularyTileSize, vocabularySize)
            let logits = outputSlice(hidden, range: start..<end)
            let values = MLX.max(logits, axis: -1)
            let ids = logits.argMax(axis: -1).asType(.int32) + start
            if let currentValues = bestValues, let currentIDs = bestIDs {
                let replace = values .> currentValues
                bestValues = which(replace, values, currentValues)
                bestIDs = which(replace, ids, currentIDs)
            } else {
                bestValues = values
                bestIDs = ids
            }
            // Keep only the running maximum and ID alive rather than a lazy
            // graph containing every vocabulary tile's logits.
            MLX.eval(bestValues!, bestIDs!)
        }
        return bestIDs!
    }

    private func encode(
        _ features: MLXArray,
        activationPrecision: GraniteActivationPrecision,
        middleVocabularyTileSize: Int = 0,
        audit: ((String, MLXArray) -> Void)?
    ) -> MLXArray {
        var hidden = inputLinear(features)
        audit?("encoder.input_projection", hidden)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, attentionDistances: attentionDistances)
            audit?("encoder.layer.\(index)", hidden)
            if index + 1 == layers.count / 2 {
                if middleVocabularyTileSize > 0,
                   middleVocabularyTileSize < output.shape.0 {
                    hidden = hidden + tiledMiddleConditioning(
                        hidden, vocabularyTileSize: middleVocabularyTileSize)
                } else {
                    let middleLogits = output(hidden)
                    audit?("encoder.middle_logits", middleLogits)
                    hidden = hidden + middleOutput(
                        softmax(middleLogits.asType(.float32), axis: -1)
                            .asType(hidden.dtype))
                }
                audit?("encoder.middle_conditioned", hidden)
            }
            // Materialize one block at a time. Without this boundary MLX keeps
            // the complete 16-layer lazy graph alive, making memory scale
            // poorly on hour-long recordings.
            hidden = GraniteReducedPrecisionStorage.roundTrip(
                hidden, mode: activationPrecision)
        }
        return hidden
    }

    /// Computes softmax(output(hidden)) followed by middleOutput without ever
    /// materializing the complete [frames, vocabulary] tensor. The first pass
    /// computes an online log-sum-exp; the second recomputes each logits tile,
    /// normalizes it, and accumulates its contribution to the projection.
    private func tiledMiddleConditioning(
        _ hidden: MLXArray, vocabularyTileSize: Int
    ) -> MLXArray {
        let vocabularySize = output.shape.0
        var globalMaximum: MLXArray?
        var exponentialSum: MLXArray?

        for start in stride(from: 0, to: vocabularySize, by: vocabularyTileSize) {
            let end = min(start + vocabularyTileSize, vocabularySize)
            let logits = outputSlice(hidden, range: start..<end).asType(.float32)
            let tileMaximum = MLX.max(logits, axis: -1)
            if let currentMaximum = globalMaximum, let currentSum = exponentialSum {
                let nextMaximum = MLX.maximum(currentMaximum, tileMaximum)
                exponentialSum = currentSum * exp(currentMaximum - nextMaximum)
                    + MLX.sum(exp(logits - nextMaximum.expandedDimensions(axis: -1)), axis: -1)
                globalMaximum = nextMaximum
            } else {
                globalMaximum = tileMaximum
                exponentialSum = MLX.sum(
                    exp(logits - tileMaximum.expandedDimensions(axis: -1)), axis: -1)
            }
            MLX.eval(globalMaximum!, exponentialSum!)
        }

        var projected: MLXArray?
        for start in stride(from: 0, to: vocabularySize, by: vocabularyTileSize) {
            let end = min(start + vocabularyTileSize, vocabularySize)
            let logits = outputSlice(hidden, range: start..<end).asType(.float32)
            let probabilities = (
                exp(logits - globalMaximum!.expandedDimensions(axis: -1))
                    / exponentialSum!.expandedDimensions(axis: -1)
            ).asType(hidden.dtype)
            let contribution = middleOutputSlice(probabilities, inputRange: start..<end)
                .asType(.float32)
            projected = projected.map { $0 + contribution } ?? contribution
            MLX.eval(projected!)
        }

        var result = projected!.asType(hidden.dtype)
        if let bias = middleOutput.bias {
            result = result + bias
        }
        return result
    }

    private func outputSlice(_ hidden: MLXArray, range: Range<Int>) -> MLXArray {
        let logits: MLXArray
        if let quantized = output as? QuantizedLinear {
            let quantizationBiases = quantized.biases.map { $0[range, 0...] }
            logits = quantizedMM(
                hidden,
                quantized.weight[range, 0...],
                scales: quantized.scales[range, 0...],
                biases: quantizationBiases,
                transpose: true,
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                mode: quantized.mode
            )
        } else {
            logits = hidden.matmul(output.weight[range, 0...].T)
        }
        if let bias = output.bias {
            return logits + bias[range]
        }
        return logits
    }

    private func middleOutputSlice(
        _ probabilities: MLXArray, inputRange: Range<Int>
    ) -> MLXArray {
        if let quantized = middleOutput as? QuantizedLinear {
            let packing = 32 / quantized.bits
            precondition(
                inputRange.lowerBound.isMultiple(of: quantized.groupSize)
                    && inputRange.upperBound.isMultiple(of: quantized.groupSize),
                "Middle CTC vocabulary tiles must align to quantization groups."
            )
            let packedRange = (inputRange.lowerBound / packing)..<(inputRange.upperBound / packing)
            let scaleRange = (inputRange.lowerBound / quantized.groupSize)..<(inputRange.upperBound / quantized.groupSize)
            return quantizedMM(
                probabilities,
                quantized.weight[0..., packedRange],
                scales: quantized.scales[0..., scaleRange],
                biases: quantized.biases.map { $0[0..., scaleRange] },
                transpose: true,
                groupSize: quantized.groupSize,
                bits: quantized.bits,
                mode: quantized.mode
            )
        }
        return probabilities.matmul(middleOutput.weight[0..., inputRange].T)
    }
}

final class GraniteCTCModel: Module, @unchecked Sendable {
    @ModuleInfo(key: "encoder") var encoder: GraniteEncoder
    init(_ config: GraniteModelConfiguration) { self._encoder.wrappedValue = GraniteEncoder(config) }
    func callAsFunction(_ features: MLXArray) -> MLXArray { encoder(features) }
    func forward(
        _ features: MLXArray,
        activationPrecision: GraniteActivationPrecision = .baseline,
        audit: ((String, MLXArray) -> Void)?
    ) -> MLXArray {
        encoder.forward(
            features, activationPrecision: activationPrecision, audit: audit)
    }
    func greedyFrameIDs(
        _ features: MLXArray,
        activationPrecision: GraniteActivationPrecision,
        vocabularyTileSize: Int,
        middleVocabularyTileSize: Int = 0
    ) -> MLXArray {
        encoder.greedyFrameIDs(
            features,
            activationPrecision: activationPrecision,
            vocabularyTileSize: vocabularyTileSize,
            middleVocabularyTileSize: middleVocabularyTileSize
        )
    }
}
