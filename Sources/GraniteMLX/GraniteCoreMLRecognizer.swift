import CoreML
import CryptoKit
import Foundation
import MLX

/// Compute-unit policy used by the Core ML speech encoder.
public enum GraniteCoreMLComputeUnits: String, Codable, Sendable, CaseIterable {
    /// Allow Core ML to choose among CPU, GPU, and Neural Engine.
    case all
    /// Keep compatible operations on CPU and GPU.
    case cpuAndGPU = "cpu-gpu"
    /// Use CPU and Neural Engine, with CPU fallback for unsupported attention operations.
    case cpuAndNeuralEngine = "cpu-ne"
    /// Run the model on CPU only.
    case cpuOnly = "cpu"

    var coreMLValue: MLComputeUnits {
        switch self {
        case .all: .all
        case .cpuAndGPU: .cpuAndGPU
        case .cpuAndNeuralEngine: .cpuAndNeuralEngine
        case .cpuOnly: .cpuOnly
        }
    }
}

/// Errors produced by the fixed-shape Core ML Granite backend.
public enum GraniteCoreMLRecognizerError: Error, GraniteDiagnosticError, Sendable {
    /// Required tokenizer or configuration files are absent.
    /// - Parameter details: Missing file, parse, or tokenizer diagnostic context.
    case invalidTokenizerDirectory(URL, details: String)
    /// The Core ML package could not be loaded or has an incompatible interface.
    /// - Parameter details: Compilation, loading, or interface diagnostic context.
    case invalidModel(URL, details: String)
    /// A chunk contains more frontend frames than the fixed model accepts.
    /// - Parameters:
    ///   - actualFrames: Frontend frames required by the supplied audio chunk.
    ///   - maximumFrames: Fixed frame capacity encoded in the Core ML graph.
    case inputTooLong(actualFrames: Int, maximumFrames: Int)
    /// Core ML prediction failed.
    /// - Parameter details: Core ML prediction or output-conversion diagnostics.
    case predictionFailed(details: String)

    /// Stable diagnostic identifier for the failure.
    public var diagnosticCode: String {
        switch self {
        case .invalidTokenizerDirectory: "GMLX-COREML-001"
        case .invalidModel: "GMLX-COREML-002"
        case .inputTooLong: "GMLX-COREML-003"
        case .predictionFailed: "GMLX-COREML-004"
        }
    }

    /// Low-level context useful for diagnostics.
    public var technicalDetails: String? {
        switch self {
        case .invalidTokenizerDirectory(let url, let details):
            "tokenizer_directory=\(url.path); \(details)"
        case .invalidModel(let url, let details):
            "model=\(url.path); \(details)"
        case .inputTooLong(let actual, let maximum):
            "actual_feature_frames=\(actual); maximum_feature_frames=\(maximum)"
        case .predictionFailed(let details): details
        }
    }

    /// User-facing localized failure description containing the diagnostic code.
    public var errorDescription: String? {
        switch self {
        case .invalidTokenizerDirectory:
            "[\(diagnosticCode)] The Granite tokenizer directory is incomplete or incompatible. Technical details: \(technicalDetails!)."
        case .invalidModel:
            "[\(diagnosticCode)] The Core ML Granite model is incomplete or incompatible. Technical details: \(technicalDetails!)."
        case .inputTooLong:
            "[\(diagnosticCode)] The audio chunk is too long for this fixed-shape Core ML model. Reduce the chunk duration. Technical details: \(technicalDetails!)."
        case .predictionFailed:
            "[\(diagnosticCode)] Core ML speech inference failed. Technical details: \(technicalDetails!)."
        }
    }
}

/// Timing breakdown for the most recent Core ML transcription.
public struct GraniteCoreMLPerformance: Codable, Sendable, Equatable {
    /// Number of fixed-shape model invocations.
    public internal(set) var chunkCount = 0
    /// Time spent constructing and evaluating Granite frontend features.
    public internal(set) var featureExtractionSeconds = 0.0
    /// Time spent allocating and copying the fixed Core ML input tensor.
    public internal(set) var inputCopySeconds = 0.0
    /// Time spent inside Core ML prediction calls.
    public internal(set) var predictionSeconds = 0.0
    /// Individual Core ML prediction durations, in invocation order.
    public internal(set) var predictionDurations: [Double] = []
    /// Time spent copying CTC frame IDs out of Core ML.
    public internal(set) var outputCopySeconds = 0.0

    /// Creates an empty timing breakdown.
    public init() {}
}

/// Granite Speech recognizer backed by a fixed-shape Core ML ML Program.
///
/// The model package consumes the same `[1, frames, 320]` frontend tensor as
/// ``GraniteRecognizer`` and returns one greedy CTC token ID per output frame.
public final class GraniteCoreMLRecognizer: @unchecked Sendable {
    /// Default directory used for persistent, OS-specific compiled Core ML models.
    /// This compatibility property follows ``GraniteModelStorage/default``.
    public static var defaultCompiledModelCacheURL: URL {
        GraniteModelStorage.default.compiledCoreMLDirectory
            ?? FileManager.default.temporaryDirectory.appendingPathComponent(
                "GraniteMLX/CoreML", isDirectory: true)
    }

    /// Core ML model package used for speech inference.
    public let modelURL: URL
    /// Directory containing Granite's `config.json` and tokenizer files.
    public let tokenizerURL: URL
    /// Decoded Granite architecture configuration.
    public let configuration: GraniteModelConfiguration
    /// Frontend used to create Granite log-mel features.
    public let featureExtractor: GraniteFeatureExtractor
    /// Fixed number of frontend frames accepted by the Core ML package.
    public let featureFrameCount: Int
    /// Audio duration represented by one fixed input, in seconds.
    public var maximumAudioDuration: Double { Double(featureFrameCount) / 50 }
    /// Timing breakdown for the most recent completed or failed transcription.
    public private(set) var lastPerformance = GraniteCoreMLPerformance()

    private let model: MLModel
    private let tokenizer: GraniteTokenizer
    private let temporaryCompiledModelURL: URL?

    /// Loads a converted Core ML package and the matching Granite tokenizer.
    /// - Parameters:
    ///   - modelURL: Compiled `.mlmodelc` or source `.mlpackage` containing the
    ///     fixed-shape Granite ML Program.
    ///   - tokenizerURL: Directory containing matching `config.json` and
    ///     tokenizer files.
    ///   - computeUnits: Core ML device-placement policy.
    ///   - compiledModelCacheURL: Persistent cache for compiled `.mlpackage`
    ///     output. Pass `nil` to use a temporary compiled model removed at
    ///     recognizer deinitialization.
    /// - Throws: ``GraniteCoreMLRecognizerError`` when tokenizer files are
    ///   incompatible, compilation/loading fails, or model inputs and outputs do
    ///   not match Granite's fixed interface.
    public init(
        modelURL: URL,
        tokenizerURL: URL,
        computeUnits: GraniteCoreMLComputeUnits = .cpuAndGPU,
        compiledModelCacheURL: URL? = GraniteCoreMLRecognizer.defaultCompiledModelCacheURL
    ) throws {
        let configURL = tokenizerURL.appendingPathComponent("config.json")
        do {
            configuration = try JSONDecoder().decode(
                GraniteModelConfiguration.self, from: Data(contentsOf: configURL))
            tokenizer = try GraniteTokenizer(directory: tokenizerURL)
        } catch {
            throw GraniteCoreMLRecognizerError.invalidTokenizerDirectory(
                tokenizerURL, details: "underlying=\(String(reflecting: error))")
        }

        let model: MLModel
        do {
            let modelConfiguration = MLModelConfiguration()
            modelConfiguration.computeUnits = computeUnits.coreMLValue
            if #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *) {
                modelConfiguration.optimizationHints.specializationStrategy = .fastPrediction
            }
            let loadURL: URL
            if modelURL.pathExtension == "mlmodelc" {
                loadURL = modelURL
                temporaryCompiledModelURL = nil
            } else {
                loadURL = try Self.compileModel(
                    at: modelURL, cacheDirectory: compiledModelCacheURL)
                temporaryCompiledModelURL = compiledModelCacheURL == nil ? loadURL : nil
            }
            model = try MLModel(contentsOf: loadURL, configuration: modelConfiguration)
        } catch {
            throw GraniteCoreMLRecognizerError.invalidModel(
                modelURL, details: "load_error=\(String(reflecting: error))")
        }
        guard let constraint = model.modelDescription
            .inputDescriptionsByName["features"]?.multiArrayConstraint,
              constraint.shape.count == 3,
              constraint.shape[0].intValue == 1,
              constraint.shape[2].intValue == 320,
              constraint.shape[1].intValue > 0,
              let output = model.modelDescription
                .outputDescriptionsByName["frame_ids"]?.multiArrayConstraint,
              output.dataType == .int32 else {
            throw GraniteCoreMLRecognizerError.invalidModel(
                modelURL,
                details: "expected_input=features[1,frames,320]/float16; expected_output=frame_ids/int32")
        }

        self.modelURL = modelURL
        self.tokenizerURL = tokenizerURL
        self.model = model
        self.featureFrameCount = constraint.shape[1].intValue
        self.featureExtractor = GraniteFeatureExtractor(sampleRate: configuration.sampleRate)
    }

    /// Downloads or loads a published Core ML Granite model and initializes it.
    ///
    /// - Parameters:
    ///   - modelSource: Catalog alias, Hugging Face repository ID, or local
    ///     repository directory containing `coreml_config.json`.
    ///   - storage: Explicit model, transfer-cache, and compiled-model locations.
    ///   - computeUnits: Core ML device-placement policy.
    ///   - hfToken: Optional Hugging Face token for private or gated models.
    ///   - progressHandler: Optional model-download progress callback.
    ///   - cancellationToken: Optional cooperative cancellation token.
    /// - Throws: Errors from ``GraniteCoreMLModelLoader`` or the designated
    ///   initializer when acquisition, validation, compilation, or loading fails.
    public convenience init(
        modelSource: String = GraniteCoreMLModelLoader.defaultModelID,
        storage: GraniteModelStorage = .default,
        computeUnits: GraniteCoreMLComputeUnits = .cpuAndGPU,
        hfToken: String? = nil,
        progressHandler: GraniteModelDownloadProgressHandler? = nil,
        cancellationToken: GraniteCancellationToken? = nil
    ) throws {
        let artifact = try GraniteCoreMLModelLoader.load(
            source: modelSource,
            storage: storage,
            hfToken: hfToken,
            progressHandler: progressHandler,
            cancellationToken: cancellationToken)
        try self.init(
            artifact: artifact,
            computeUnits: computeUnits,
            compiledModelCacheURL: storage.compiledCoreMLDirectory)
    }

    /// Downloads or loads a Core ML model while overriding compiled-model storage.
    /// - Parameters:
    ///   - modelSource: Catalog alias, Hugging Face repository ID, or local repository.
    ///   - storage: Explicit model and transfer-cache locations.
    ///   - computeUnits: Core ML device-placement policy.
    ///   - hfToken: Optional Hugging Face token for private or gated models.
    ///   - compiledModelCacheURL: Persistent compilation directory, or `nil` for
    ///     temporary compilation output.
    ///   - progressHandler: Optional model-download progress callback.
    ///   - cancellationToken: Optional cooperative cancellation token.
    /// - Throws: Model acquisition, validation, compilation, and loading errors.
    public convenience init(
        modelSource: String = GraniteCoreMLModelLoader.defaultModelID,
        storage: GraniteModelStorage = .default,
        computeUnits: GraniteCoreMLComputeUnits = .cpuAndGPU,
        hfToken: String? = nil,
        compiledModelCacheURL: URL?,
        progressHandler: GraniteModelDownloadProgressHandler? = nil,
        cancellationToken: GraniteCancellationToken? = nil
    ) throws {
        let artifact = try GraniteCoreMLModelLoader.load(
            source: modelSource,
            storage: storage,
            hfToken: hfToken,
            progressHandler: progressHandler,
            cancellationToken: cancellationToken)
        try self.init(
            artifact: artifact,
            computeUnits: computeUnits,
            compiledModelCacheURL: compiledModelCacheURL)
    }

    /// Initializes a recognizer from a validated Core ML model artifact.
    /// - Parameters:
    ///   - artifact: Validated package, tokenizer directory, and configuration.
    ///   - computeUnits: Core ML device-placement policy.
    ///   - compiledModelCacheURL: Persistent compilation cache, or `nil` for a
    ///     temporary compiled model.
    /// - Throws: ``GraniteCoreMLRecognizerError`` when tokenizer loading,
    ///   compilation, model loading, or interface validation fails.
    public convenience init(
        artifact: GraniteCoreMLModelArtifact,
        computeUnits: GraniteCoreMLComputeUnits = .cpuAndGPU,
        compiledModelCacheURL: URL? = GraniteCoreMLRecognizer.defaultCompiledModelCacheURL
    ) throws {
        try self.init(
            modelURL: artifact.modelURL,
            tokenizerURL: artifact.tokenizerURL,
            computeUnits: computeUnits,
            compiledModelCacheURL: compiledModelCacheURL)
    }

    deinit {
        if let temporaryCompiledModelURL {
            try? FileManager.default.removeItem(at: temporaryCompiledModelURL)
        }
    }

    /// Exposes the shared Granite frontend tensor for parity diagnostics.
    /// - Parameter audio: Prepared mono audio. Callers are responsible for using
    ///   the sample rate in ``configuration``.
    /// - Returns: Granite frontend features shaped `[1, frames, 320]`.
    public func features(for audio: GraniteAudio) -> MLXArray {
        featureExtractor(audio.samples)
    }

    /// Transcribes prepared audio using fixed-shape, context-overlapped chunks.
    ///
    /// `audioChunkDuration + 2 * audioChunkContext` must not exceed the model's
    /// ``maximumAudioDuration``. When either value is `nil`, the recognizer
    /// chooses up to 20.48 seconds of context per side and the largest central
    /// chunk that fits. Set the duration to zero only when the complete input
    /// fits in one invocation.
    ///
    /// - Parameters:
    ///   - audio: Mono audio at the checkpoint's required sample rate, normally
    ///     produced by ``GraniteAudioInput/load(url:targetSampleRate:cancellationToken:progressHandler:)``.
    ///   - audioChunkDuration: Seconds of central, non-overlapping audio retained
    ///     from each invocation. `nil` chooses the largest core that fits the
    ///     fixed Core ML input after reserving context. `0` disables chunking and
    ///     therefore requires the complete input to fit the model graph.
    ///   - audioChunkContext: Seconds of overlapping audio evaluated on each side
    ///     of a central chunk and discarded from its output. `nil` chooses up to
    ///     20.48 seconds per side. More context improves boundary information but
    ///     leaves less graph capacity for central audio.
    ///   - cancellationToken: Optional cooperative cancellation token, checked
    ///     before predictions and between chunks. An active Core ML prediction
    ///     cannot stop immediately.
    ///   - progressHandler: Receives synchronous phase and chunk progress updates
    ///     on the thread performing transcription.
    /// - Returns: Raw text plus collapsed CTC token timing, approximate word
    ///   timing, source duration, and model metadata.
    /// - Throws: ``GraniteAudioError`` for incompatible audio,
    ///   ``GraniteCoreMLRecognizerError`` when the selected graph cannot hold the
    ///   requested input/profile, and ``GraniteOperationError`` for cancellation
    ///   or wrapped framework failures.
    public func transcribe(
        _ audio: GraniteAudio,
        audioChunkDuration: Double? = nil,
        audioChunkContext: Double? = nil,
        cancellationToken: GraniteCancellationToken? = nil,
        progressHandler: GraniteOperationProgressHandler? = nil
    ) throws -> GraniteTranscription {
        lastPerformance = GraniteCoreMLPerformance()
        try cancellationToken?.checkCancellation(operation: "Core ML speech transcription")
        guard audio.sampleRate == configuration.sampleRate else {
            throw GraniteAudioError.invalidAudioFormat(
                details: "prepared_sample_rate=\(audio.sampleRate); required_sample_rate=\(configuration.sampleRate); source=\(audio.source.path)")
        }
        guard audio.samples.count > 256 else {
            throw GraniteAudioError.invalidAudioFormat(
                details: "prepared_sample_count=\(audio.samples.count); minimum_sample_count=257; source=\(audio.source.path)")
        }
        let effectiveChunkContext = audioChunkContext
            ?? min(20.48, maximumAudioDuration / 4)
        let effectiveChunkDuration = audioChunkDuration
            ?? max(0, maximumAudioDuration - 2 * effectiveChunkContext)
        guard effectiveChunkDuration >= 0, effectiveChunkContext >= 0 else {
            throw GraniteRecognizerError.unsupportedConfiguration(
                "Chunk duration and context must be non-negative; chunk_duration=\(effectiveChunkDuration), chunk_context=\(effectiveChunkContext).")
        }
        if effectiveChunkDuration > 0,
           effectiveChunkDuration + 2 * effectiveChunkContext > maximumAudioDuration + 0.0001 {
            throw GraniteRecognizerError.unsupportedConfiguration(
                "Core ML chunk plus context exceeds the fixed model input; chunk_duration=\(effectiveChunkDuration), context=\(effectiveChunkContext), model_maximum=\(maximumAudioDuration).")
        }

        let frameIDs: [Int]
        if effectiveChunkDuration > 0, audio.duration > effectiveChunkDuration {
            frameIDs = try transcribeChunks(
                audio,
                coreDuration: effectiveChunkDuration,
                contextDuration: effectiveChunkContext,
                cancellationToken: cancellationToken,
                progressHandler: progressHandler)
        } else {
            guard audio.duration <= maximumAudioDuration + 0.0001 else {
                throw GraniteCoreMLRecognizerError.inputTooLong(
                    actualFrames: Int(ceil(audio.duration * 50)),
                    maximumFrames: featureFrameCount)
            }
            progressHandler?(GraniteOperationProgress(
                phase: .transcribing, fractionCompleted: 0,
                message: "Transcribing audio with Core ML", chunkIndex: 0, chunkCount: 1))
            frameIDs = try greedyFrameIDs(for: audio)
        }

        try cancellationToken?.checkCancellation(operation: "Core ML speech transcription")
        let tokenIDs = GraniteCTCDecoder.collapse(frameIDs)
        let tokenTimings = GraniteCTCDecoder.tokenTimings(
            frameIDs,
            frameRate: configuration.outputFrameRate,
            decodeToken: tokenizer.decodeToken)
        let transcription = GraniteTranscription(
            text: tokenizer.decode(tokenIDs),
            tokens: tokenTimings,
            words: GraniteCTCDecoder.words(from: tokenTimings),
            duration: audio.duration,
            model: modelURL.path)
        progressHandler?(GraniteOperationProgress(
            phase: .complete, fractionCompleted: 1,
            message: "Core ML transcription complete"))
        return transcription
    }

    private func transcribeChunks(
        _ audio: GraniteAudio,
        coreDuration: Double,
        contextDuration: Double,
        cancellationToken: GraniteCancellationToken?,
        progressHandler: GraniteOperationProgressHandler?
    ) throws -> [Int] {
        let chunkSamples = max(1, Int(coreDuration * Double(audio.sampleRate)))
        let contextSamples = max(0, Int(contextDuration * Double(audio.sampleRate)))
        let starts = Array(stride(from: 0, to: audio.samples.count, by: chunkSamples))
        var result: [Int] = []
        for (index, coreStart) in starts.enumerated() {
            try cancellationToken?.checkCancellation(operation: "Core ML speech transcription")
            progressHandler?(GraniteOperationProgress(
                phase: .transcribing,
                fractionCompleted: Double(index) / Double(starts.count),
                message: "Transcribing Core ML chunk \(index + 1) of \(starts.count)",
                chunkIndex: index, chunkCount: starts.count))
            let coreEnd = min(coreStart + chunkSamples, audio.samples.count)
            let segmentStart = max(0, coreStart - contextSamples)
            let segmentEnd = min(audio.samples.count, coreEnd + contextSamples)
            let segment = GraniteAudio(
                samples: Array(audio.samples[segmentStart..<segmentEnd]),
                sampleRate: audio.sampleRate,
                source: audio.source)
            let segmentIDs = try greedyFrameIDs(for: segment)
            let leadingSeconds = Double(coreStart - segmentStart) / Double(audio.sampleRate)
            let coreSeconds = Double(coreEnd - coreStart) / Double(audio.sampleRate)
            let first = min(
                segmentIDs.count,
                Int((leadingSeconds * configuration.outputFrameRate).rounded()))
            let end = min(
                segmentIDs.count,
                first + Int((coreSeconds * configuration.outputFrameRate).rounded()))
            result.append(contentsOf: segmentIDs[first..<end])
        }
        return result
    }

    private func greedyFrameIDs(for audio: GraniteAudio) throws -> [Int] {
        try autoreleasepool {
            try greedyFrameIDsInsideAutoreleasePool(for: audio)
        }
    }

    private func greedyFrameIDsInsideAutoreleasePool(for audio: GraniteAudio) throws -> [Int] {
        lastPerformance.chunkCount += 1
        let featureStart = ProcessInfo.processInfo.systemUptime
        let features = features(for: audio).asType(.float16)
        MLX.eval(features)
        lastPerformance.featureExtractionSeconds +=
            ProcessInfo.processInfo.systemUptime - featureStart
        let shape = features.shape
        guard shape.count == 3, shape[0] == 1, shape[2] == 320 else {
            throw GraniteCoreMLRecognizerError.predictionFailed(
                details: "unexpected_frontend_shape=\(shape)")
        }
        guard shape[1] <= featureFrameCount else {
            throw GraniteCoreMLRecognizerError.inputTooLong(
                actualFrames: shape[1], maximumFrames: featureFrameCount)
        }

        do {
            let inputStart = ProcessInfo.processInfo.systemUptime
            let input = try MLMultiArray(
                shape: [1, NSNumber(value: featureFrameCount), 320],
                dataType: .float16)
            let featureData = features.asData()
            let inputBytes = input.count * MemoryLayout<Float16>.stride
            input.dataPointer.initializeMemory(
                as: UInt8.self, repeating: 0, count: inputBytes)
            featureData.data.withUnsafeBytes { source in
                guard let baseAddress = source.baseAddress else { return }
                input.dataPointer.copyMemory(
                    from: baseAddress,
                    byteCount: min(source.count, inputBytes))
            }
            lastPerformance.inputCopySeconds +=
                ProcessInfo.processInfo.systemUptime - inputStart
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "features": MLFeatureValue(multiArray: input)
            ])
            let predictionStart = ProcessInfo.processInfo.systemUptime
            let prediction = try model.prediction(from: provider)
            let predictionDuration = ProcessInfo.processInfo.systemUptime - predictionStart
            lastPerformance.predictionSeconds += predictionDuration
            lastPerformance.predictionDurations.append(predictionDuration)
            let outputStart = ProcessInfo.processInfo.systemUptime
            guard let output = prediction.featureValue(for: "frame_ids")?.multiArrayValue,
                  output.dataType == .int32 else {
                throw GraniteCoreMLRecognizerError.predictionFailed(
                    details: "missing_or_invalid_output=frame_ids")
            }
            let actualFrames = min(output.count, shape[1] / 4)
            let outputPointer = output.dataPointer.bindMemory(
                to: Int32.self, capacity: output.count)
            let result = (0..<actualFrames).map { Int(outputPointer[$0]) }
            lastPerformance.outputCopySeconds +=
                ProcessInfo.processInfo.systemUptime - outputStart
            return result
        } catch let error as GraniteCoreMLRecognizerError {
            throw error
        } catch {
            throw GraniteCoreMLRecognizerError.predictionFailed(
                details: "underlying=\(String(reflecting: error)); feature_frames=\(shape[1]); fixed_frames=\(featureFrameCount)")
        }
    }

    private static func compileModel(at url: URL, cacheDirectory: URL?) throws -> URL {
        let cachedURL: URL?
        if let cacheDirectory {
            try FileManager.default.createDirectory(
                at: cacheDirectory, withIntermediateDirectories: true)
            cachedURL = try compiledModelCacheURL(
                for: url, cacheDirectory: cacheDirectory)
            if FileManager.default.fileExists(atPath: cachedURL!.path) {
                return cachedURL!
            }
        } else {
            cachedURL = nil
        }

        let semaphore = DispatchSemaphore(value: 0)
        let lock = NSLock()
        var result: Result<URL, Error>?
        MLModel.compileModel(at: url) { compilationResult in
            lock.lock()
            result = compilationResult
            lock.unlock()
            semaphore.signal()
        }
        semaphore.wait()
        lock.lock()
        defer { lock.unlock() }
        let compiled = try result!.get()
        guard let cachedURL else { return compiled }
        do {
            try FileManager.default.moveItem(at: compiled, to: cachedURL)
            return cachedURL
        } catch CocoaError.fileWriteFileExists {
            try? FileManager.default.removeItem(at: compiled)
            return cachedURL
        } catch {
            try? FileManager.default.removeItem(at: compiled)
            throw error
        }
    }

    static func compiledModelCacheSize(
        for modelURL: URL,
        cacheDirectory: URL = GraniteCoreMLRecognizer.defaultCompiledModelCacheURL
    ) -> Int64 {
        guard let url = try? compiledModelCacheURL(
            for: modelURL, cacheDirectory: cacheDirectory),
              FileManager.default.fileExists(atPath: url.path) else { return 0 }
        return GraniteModelCache.directorySize(url)
    }

    @discardableResult
    static func removeCompiledModelCache(
        for modelURL: URL,
        cacheDirectory: URL = GraniteCoreMLRecognizer.defaultCompiledModelCacheURL
    ) throws -> Int64 {
        let url = try compiledModelCacheURL(
            for: modelURL, cacheDirectory: cacheDirectory)
        guard FileManager.default.fileExists(atPath: url.path) else { return 0 }
        let bytes = GraniteModelCache.directorySize(url)
        try FileManager.default.removeItem(at: url)
        return bytes
    }

    private static func compiledModelCacheURL(
        for modelURL: URL, cacheDirectory: URL
    ) throws -> URL {
        cacheDirectory
            .appendingPathComponent(try compilationCacheKey(for: modelURL))
            .appendingPathExtension("mlmodelc")
    }

    private static func compilationCacheKey(for modelURL: URL) throws -> String {
        let fileManager = FileManager.default
        let keys: Set<URLResourceKey> = [.fileSizeKey, .contentModificationDateKey]
        guard let enumerator = fileManager.enumerator(
            at: modelURL,
            includingPropertiesForKeys: Array(keys),
            options: [.skipsHiddenFiles]) else {
            throw GraniteCoreMLRecognizerError.invalidModel(
                modelURL, details: "unable_to_enumerate_model_package")
        }
        var entries: [String] = []
        for case let fileURL as URL in enumerator {
            let values = try fileURL.resourceValues(forKeys: keys)
            guard let size = values.fileSize else { continue }
            let relative = String(fileURL.path.dropFirst(modelURL.path.count))
            let modified = values.contentModificationDate?.timeIntervalSince1970 ?? 0
            entries.append("\(relative)|\(size)|\(modified)")
        }
        entries.sort()
        let identity = ([
            "granite-coreml-cache-v1",
            ProcessInfo.processInfo.operatingSystemVersionString,
        ] + entries).joined(separator: "\n")
        return SHA256.hash(data: Data(identity.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
    }
}
