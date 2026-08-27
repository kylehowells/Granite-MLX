@preconcurrency import AVFoundation
import Foundation

public struct GraniteAudio: Sendable {
    public let samples: [Float]
    public let sampleRate: Int
    public let source: URL

    public var duration: Double {
        guard sampleRate > 0 else { return 0 }
        return Double(samples.count) / Double(sampleRate)
    }
}

public enum GraniteAudioError: Error, LocalizedError {
    case noAudioTrack(URL)
    case invalidAudioFormat
    case conversionFailed(String)
    case ffmpegUnavailable
    case ffmpegFailed(String)

    public var errorDescription: String? {
        switch self {
        case .noAudioTrack(let url): "No readable audio track found in \(url.path)."
        case .invalidAudioFormat: "The input audio format could not be read."
        case .conversionFailed(let message): "Audio conversion failed: \(message)"
        case .ffmpegUnavailable: "AVFoundation could not read this file and ffmpeg was not found."
        case .ffmpegFailed(let message): "ffmpeg failed: \(message)"
        }
    }
}

public enum GraniteAudioInput {
    public static let targetSampleRate = 16_000

    /// Loads ordinary audio containers with AVFoundation, then falls back to
    /// ffmpeg for video and less common formats supported by the local install.
    public static func load(url: URL, targetSampleRate: Int = targetSampleRate) throws -> GraniteAudio {
        do {
            return try loadWithAVFoundation(url: url, targetSampleRate: targetSampleRate)
        } catch {
            return try loadWithFFmpeg(url: url, targetSampleRate: targetSampleRate)
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
            throw GraniteAudioError.invalidAudioFormat
        }
        try file.read(into: input)
        guard let channelData = input.floatChannelData else {
            throw GraniteAudioError.invalidAudioFormat
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

    private static func loadWithFFmpeg(url: URL, targetSampleRate: Int) throws -> GraniteAudio {
        guard FileManager.default.isExecutableFile(atPath: "/opt/homebrew/bin/ffmpeg") || FileManager.default.isExecutableFile(atPath: "/usr/local/bin/ffmpeg") || FileManager.default.isExecutableFile(atPath: "/usr/bin/ffmpeg") else {
            throw GraniteAudioError.ffmpegUnavailable
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", url.path, "-f", "f32le", "-ac", "1", "-ar", String(targetSampleRate), "pipe:1"]
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        let data = stdout.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            let message = String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "unknown ffmpeg error"
            throw GraniteAudioError.ffmpegFailed(message)
        }
        guard data.count >= MemoryLayout<Float>.size else { throw GraniteAudioError.noAudioTrack(url) }
        let samples = data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }
        return GraniteAudio(samples: samples, sampleRate: targetSampleRate, source: url)
    }
}
