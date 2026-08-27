@preconcurrency import AVFoundation
import Foundation

/// Mono floating-point audio prepared for Granite speech recognition.
public struct GraniteAudio: Sendable {
    /// Normalized mono samples.
    public let samples: [Float]
    /// Samples per second.
    public let sampleRate: Int
    /// Original media URL.
    public let source: URL

    /// Audio duration in seconds.
    public var duration: Double {
        guard sampleRate > 0 else { return 0 }
        return Double(samples.count) / Double(sampleRate)
    }

    /// Creates an in-memory audio value.
    public init(samples: [Float], sampleRate: Int, source: URL) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.source = source
    }
}

/// Errors produced while decoding and converting input media.
public enum GraniteAudioError: Error, GraniteDiagnosticError {
    /// The input path does not identify a regular file.
    case inputNotFound(URL)
    /// No usable audio track was found.
    case noAudioTrack(URL)
    /// AVFoundation returned an unsupported audio representation.
    case invalidAudioFormat(details: String)
    /// Audio conversion or resampling failed.
    case conversionFailed(String)
    /// Direct decoding failed and ffmpeg is not installed.
    case ffmpegUnavailable(URL, directDecoderDetails: String)
    /// ffmpeg could not be launched.
    case ffmpegLaunchFailed(URL, details: String)
    /// ffmpeg ran but rejected or failed to decode the input.
    case ffmpegFailed(URL, exitStatus: Int32, stderr: String)

    /// Stable diagnostic identifier for the failure.
    public var diagnosticCode: String {
        switch self {
        case .inputNotFound: "GMLX-AUDIO-001"
        case .noAudioTrack: "GMLX-AUDIO-002"
        case .invalidAudioFormat: "GMLX-AUDIO-003"
        case .conversionFailed: "GMLX-AUDIO-004"
        case .ffmpegUnavailable: "GMLX-AUDIO-005"
        case .ffmpegLaunchFailed: "GMLX-AUDIO-006"
        case .ffmpegFailed: "GMLX-AUDIO-007"
        }
    }

    /// Low-level context useful for diagnostics.
    public var technicalDetails: String? {
        switch self {
        case .inputNotFound(let url), .noAudioTrack(let url): "path=\(url.path)"
        case .invalidAudioFormat(let details), .conversionFailed(let details): details
        case .ffmpegUnavailable(let url, let details):
            "path=\(url.path); direct_decoder=\(details); PATH=\(ProcessInfo.processInfo.environment["PATH"] ?? "<unset>")"
        case .ffmpegLaunchFailed(let url, let details): "path=\(url.path); launch_error=\(details)"
        case .ffmpegFailed(let url, let status, let stderr):
            "path=\(url.path); exit_status=\(status); stderr=\(stderr.trimmingCharacters(in: .whitespacesAndNewlines))"
        }
    }

    /// User-facing localized failure description containing the diagnostic code.
    public var errorDescription: String? {
        switch self {
        case .inputNotFound(let url):
            "[\(diagnosticCode)] Input file was not found: \(url.path). Technical details: \(technicalDetails!)."
        case .noAudioTrack(let url):
            "[\(diagnosticCode)] No readable audio track was found in \(url.lastPathComponent). Technical details: \(technicalDetails!)."
        case .invalidAudioFormat:
            "[\(diagnosticCode)] The direct audio decoder returned an unsupported format. Technical details: \(technicalDetails!)."
        case .conversionFailed:
            "[\(diagnosticCode)] Audio resampling or conversion failed. Technical details: \(technicalDetails!)."
        case .ffmpegUnavailable(let url, _):
            "[\(diagnosticCode)] \(url.lastPathComponent) could not be read directly and ffmpeg is not installed. Install it with `brew install ffmpeg`, or provide a directly readable audio file such as WAV. Technical details: \(technicalDetails!)."
        case .ffmpegLaunchFailed:
            "[\(diagnosticCode)] ffmpeg was found but could not be launched. Technical details: \(technicalDetails!)."
        case .ffmpegFailed:
            "[\(diagnosticCode)] ffmpeg could not decode the input media. Technical details: \(technicalDetails!)."
        }
    }
}

/// Decodes local audio or video input into Granite's required audio format.
public enum GraniteAudioInput {
    /// Default sample rate required by Granite Speech 5.0.
    public static let targetSampleRate = 16_000

    /// Loads ordinary audio containers with AVFoundation, then falls back to
    /// ffmpeg for video and less common formats supported by the local install.
    /// - Parameters:
    ///   - url: Local media file to decode.
    ///   - targetSampleRate: Desired output sample rate.
    ///   - cancellationToken: Optional cooperative cancellation token.
    ///   - progressHandler: Optional application-facing progress callback.
    /// - Returns: Mono floating-point audio at `targetSampleRate`.
    public static func load(
        url: URL,
        targetSampleRate: Int = targetSampleRate,
        cancellationToken: GraniteCancellationToken? = nil,
        progressHandler: GraniteOperationProgressHandler? = nil
    ) throws -> GraniteAudio {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GraniteAudioError.inputNotFound(url)
        }
        try cancellationToken?.checkCancellation(operation: "Audio loading")
        progressHandler?(GraniteOperationProgress(
            phase: .loadingAudio, fractionCompleted: 0,
            message: "Loading \(url.lastPathComponent)"))
        do {
            let audio = try loadWithAVFoundation(url: url, targetSampleRate: targetSampleRate)
            try cancellationToken?.checkCancellation(operation: "Audio loading")
            progressHandler?(GraniteOperationProgress(
                phase: .loadingAudio, fractionCompleted: 1,
                message: "Loaded \(url.lastPathComponent)"))
            return audio
        } catch let directError {
            try cancellationToken?.checkCancellation(operation: "Audio loading")
            let audio = try loadWithFFmpeg(
                url: url, targetSampleRate: targetSampleRate,
                directDecoderDetails: String(describing: directError),
                cancellationToken: cancellationToken)
            progressHandler?(GraniteOperationProgress(
                phase: .loadingAudio, fractionCompleted: 1,
                message: "Loaded \(url.lastPathComponent) with ffmpeg"))
            return audio
        }
    }

    private static func loadWithAVFoundation(url: URL, targetSampleRate: Int) throws -> GraniteAudio {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw GraniteAudioError.noAudioTrack(url)
        }
        let sourceFormat = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard let input = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: frameCount) else {
            throw GraniteAudioError.invalidAudioFormat(details: "AVAudioPCMBuffer allocation failed; format=\(sourceFormat)")
        }
        try file.read(into: input)
        guard let channelData = input.floatChannelData else {
            throw GraniteAudioError.invalidAudioFormat(details: "AVAudioPCMBuffer did not expose float channel data; format=\(sourceFormat)")
        }
        let channelCount = Int(sourceFormat.channelCount)
        let frameLength = Int(input.frameLength)
        var mono = [Float](repeating: 0, count: frameLength)
        for channel in 0..<channelCount {
            let samples = UnsafeBufferPointer(start: channelData[channel], count: frameLength)
            for index in 0..<frameLength { mono[index] += samples[index] / Float(channelCount) }
        }
        let sourceRate = Int(sourceFormat.sampleRate.rounded())
        let output = sourceRate == targetSampleRate
            ? mono
            : try resample(mono, from: sourceRate, to: targetSampleRate)
        return GraniteAudio(samples: output, sampleRate: targetSampleRate, source: url)
    }

    private static func resample(_ input: [Float], from sourceRate: Int, to targetRate: Int) throws -> [Float] {
        guard let inputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sourceRate), channels: 1, interleaved: false),
              let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(targetRate), channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inputFormat, to: outputFormat),
              let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: AVAudioFrameCount(input.count)) else {
            throw GraniteAudioError.conversionFailed("unable to create resampler")
        }
        inputBuffer.frameLength = AVAudioFrameCount(input.count)
        input.withUnsafeBufferPointer { pointer in
            guard let base = pointer.baseAddress else { return }
            inputBuffer.floatChannelData![0].update(from: base, count: input.count)
        }
        let ratio = Double(targetRate) / Double(sourceRate)
        let capacity = AVAudioFrameCount(ceil(Double(input.count) * ratio) + 64)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else {
            throw GraniteAudioError.conversionFailed("unable to allocate resampler output")
        }
        final class InputProvider: @unchecked Sendable {
            let buffer: AVAudioPCMBuffer
            var consumed = false
            init(buffer: AVAudioPCMBuffer) { self.buffer = buffer }
        }
        let provider = InputProvider(buffer: inputBuffer)
        var conversionError: NSError?
        converter.convert(to: outputBuffer, error: &conversionError) { _, status in
            if provider.consumed {
                status.pointee = .endOfStream
                return nil
            }
            provider.consumed = true
            status.pointee = .haveData
            return provider.buffer
        }
        if let conversionError { throw GraniteAudioError.conversionFailed(conversionError.localizedDescription) }
        guard let outputData = outputBuffer.floatChannelData?[0] else { throw GraniteAudioError.conversionFailed("no output samples") }
        return Array(UnsafeBufferPointer(start: outputData, count: Int(outputBuffer.frameLength)))
    }

    private static func loadWithFFmpeg(
        url: URL,
        targetSampleRate: Int,
        directDecoderDetails: String,
        cancellationToken: GraniteCancellationToken?
    ) throws -> GraniteAudio {
        guard let executable = ffmpegExecutable() else {
            throw GraniteAudioError.ffmpegUnavailable(
                url, directDecoderDetails: directDecoderDetails)
        }
        let process = Process()
        process.executableURL = executable
        process.arguments = ["-hide_banner", "-loglevel", "error", "-i", url.path, "-f", "f32le", "-ac", "1", "-ar", String(targetSampleRate), "pipe:1"]
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        do { try process.run() }
        catch { throw GraniteAudioError.ffmpegLaunchFailed(url, details: String(describing: error)) }
        let stdoutReader = PipeReader(handle: stdout.fileHandleForReading)
        let stderrReader = PipeReader(handle: stderr.fileHandleForReading)
        stdoutReader.start()
        stderrReader.start()
        while process.isRunning {
            if cancellationToken?.isCancelled == true {
                process.terminate()
            }
            Thread.sleep(forTimeInterval: 0.05)
        }
        process.waitUntilExit()
        let data = stdoutReader.waitForData()
        let errorData = stderrReader.waitForData()
        try cancellationToken?.checkCancellation(operation: "ffmpeg audio decoding")
        guard process.terminationStatus == 0 else {
            let message = String(data: errorData, encoding: .utf8) ?? "unknown ffmpeg error"
            throw GraniteAudioError.ffmpegFailed(
                url, exitStatus: process.terminationStatus, stderr: message)
        }
        guard data.count >= MemoryLayout<Float>.size else { throw GraniteAudioError.noAudioTrack(url) }
        let samples = data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
        return GraniteAudio(samples: samples, sampleRate: targetSampleRate, source: url)
    }

    private static func ffmpegExecutable() -> URL? {
        let manager = FileManager.default
        let searchPath = ProcessInfo.processInfo.environment["PATH"] ?? ""
        for directory in searchPath.split(separator: ":") {
            let candidate = URL(fileURLWithPath: String(directory)).appendingPathComponent("ffmpeg")
            if manager.isExecutableFile(atPath: candidate.path) { return candidate }
        }
        return nil
    }

    private final class PipeReader: @unchecked Sendable {
        private let handle: FileHandle
        private let semaphore = DispatchSemaphore(value: 0)
        private let lock = NSLock()
        private var data = Data()

        init(handle: FileHandle) {
            self.handle = handle
        }

        func start() {
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                let result = handle.readDataToEndOfFile()
                lock.lock()
                data = result
                lock.unlock()
                semaphore.signal()
            }
        }

        func waitForData() -> Data {
            semaphore.wait()
            lock.lock()
            defer { lock.unlock() }
            return data
        }
    }
}
