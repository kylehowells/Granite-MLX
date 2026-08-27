import Foundation
import MLX

/// Granite's log-mel + delta + frame-stacking frontend.
///
/// This intentionally keeps the frontend independent of the encoder so that
/// its tensors can be compared with the Python reference before the full
/// Conformer is enabled. The filterbank follows torchaudio's default HTK mel
/// scale and power spectrogram settings used by the Granite processor.
public struct GraniteFeatureExtractor: @unchecked Sendable {
    public let sampleRate: Int
    public let fftSize: Int
    public let windowLength: Int
    public let hopLength: Int
    public let melCount: Int
    public let stackFactor: Int
    public let includeDeltas: Bool
    public let deltaWindowLength: Int
    public let logMelFloorDB: Float

    private let window: MLXArray
    private let filterbank: MLXArray

    public init(
        sampleRate: Int = 16_000, fftSize: Int = 512, windowLength: Int = 400,
        hopLength: Int = 160, melCount: Int = 80, stackFactor: Int = 2,
        includeDeltas: Bool = true, deltaWindowLength: Int = 3,
        logMelFloorDB: Float = 8
    ) {
        self.sampleRate = sampleRate
        self.fftSize = fftSize
        self.windowLength = windowLength
        self.hopLength = hopLength
        self.melCount = melCount
        self.stackFactor = stackFactor
        self.includeDeltas = includeDeltas
        self.deltaWindowLength = deltaWindowLength
        self.logMelFloorDB = logMelFloorDB
        let sidePadding = (fftSize - windowLength) / 2
        self.window = MLXArray(
            [Float](repeating: 0, count: sidePadding)
                + Self.hann(count: windowLength)
                + [Float](repeating: 0, count: fftSize - windowLength - sidePadding)
        )
        let bank = Self.makeFilterbank(sampleRate: sampleRate, fftSize: fftSize, melCount: melCount)
        self.filterbank = MLXArray(bank.flatMap { $0 }).reshaped(melCount, fftSize / 2 + 1)
    }

    /// Returns [1, stackedFrames, melCount * (1 + deltas) * stackFactor].
    public func callAsFunction(_ samples: [Float]) -> MLXArray {
        let melFrames = max(1, samples.count / hopLength)
        let frameCount = max(stackFactor, ((melFrames + stackFactor - 1) / stackFactor) * stackFactor)
        let requiredSamples = (frameCount - 1) * hopLength + 1

        var signal = samples
        if signal.count < requiredSamples { signal += [Float](repeating: 0, count: requiredSamples - signal.count) }
        // torchaudio MelSpectrogram defaults to center=True and reflect padding.
        let centerPadding = fftSize / 2
        precondition(signal.count > centerPadding, "Granite audio must contain more than 256 samples")
        let prefix = Array(signal[1...centerPadding].reversed())
        let suffixStart = signal.count - centerPadding - 1
        let suffix = Array(signal[suffixStart..<(signal.count - 1)].reversed())
        signal = prefix + signal + suffix
        let audio = MLXArray(signal)
        let frames = asStrided(audio, [frameCount, fftSize], strides: [hopLength, 1], offset: 0)
        let spectrum = MLXFFT.rfft(frames * window, axis: 1)
        let power = spectrum.realPart().square() + spectrum.imaginaryPart().square()
        var mel = power.matmul(filterbank.transposed())
        mel = MLX.maximum(mel, MLXArray(Float(1e-10))).log10()
        let maximum = mel.max(axis: 0, keepDims: true).max(axis: 1, keepDims: true)
        mel = MLX.maximum(mel, maximum - logMelFloorDB) / 4 + 1

        var channels = [mel]
        if includeDeltas {
            // For the configured three-frame window, torchaudio's delta is
            // (next - previous) / 2 with edge replication.
            let previous = MLX.concatenated([mel[0..<1], mel[0..<(frameCount - 1)]], axis: 0)
            let next = MLX.concatenated([mel[1..<frameCount], mel[(frameCount - 1)..<frameCount]], axis: 0)
            channels.append((next - previous) / 2)
        }
        let features = MLX.concatenated(channels, axis: 1)
        let timeMajor = features.reshaped(frameCount / stackFactor, stackFactor * features.shape[1])
        return timeMajor.expandedDimensions(axis: 0)
    }

    private static func hann(count: Int) -> [Float] {
        (0..<count).map { 0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(count)) }
    }

    private static func makeFilterbank(sampleRate: Int, fftSize: Int, melCount: Int) -> [[Float]] {
        let bins = fftSize / 2 + 1
        func hzToMel(_ hz: Float) -> Float { 2595 * log10(1 + hz / 700) }
        func melToHz(_ mel: Float) -> Float { 700 * (pow(10, mel / 2595) - 1) }
        let points = (0..<(melCount + 2)).map { melToHz(hzToMel(0) + Float($0) * (hzToMel(Float(sampleRate) / 2) - hzToMel(0)) / Float(melCount + 1)) }
        let frequencies = (0..<bins).map { Float($0) * Float(sampleRate) / Float(fftSize) }
        return (0..<melCount).map { m in
            var row = [Float](repeating: 0, count: bins)
            for k in 0..<bins {
                let rising = (frequencies[k] - points[m]) / (points[m + 1] - points[m])
                let falling = (points[m + 2] - frequencies[k]) / (points[m + 2] - points[m + 1])
                row[k] = max(0, min(rising, falling))
            }
            return row
        }
    }
}
