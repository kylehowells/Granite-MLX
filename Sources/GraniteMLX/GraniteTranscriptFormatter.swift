import Foundation

/// Describes a transcript-formatting checkpoint independently of its runtime architecture.
public struct GraniteTranscriptFormatterInfo: Codable, Sendable, Equatable {
    /// Architecture identifier used to select a formatter backend.
    public let architecture: String
    /// Human-readable checkpoint precision.
    public let precision: String
    /// Packed weight bit width, or `nil` for floating-point weights.
    public let quantizationBits: Int?

    /// Creates formatter checkpoint metadata.
    public init(architecture: String, precision: String, quantizationBits: Int?) {
        self.architecture = architecture
        self.precision = precision
        self.quantizationBits = quantizationBits
    }
}

/// A non-ASR backend that adds presentation annotations to Granite text.
///
/// Implementations must preserve the transcript's lexical content. They may
/// add capitalization, punctuation, sentence boundaries, and other explicit
/// presentation annotations, but must not silently replace recognized words.
public protocol GraniteTranscriptFormatter: Sendable {
    /// Metadata for the loaded formatter checkpoint.
    var formatterInfo: GraniteTranscriptFormatterInfo { get }

    /// Formats raw Granite text with optional cancellation and progress reporting.
    func format(
        _ text: String,
        cancellationToken: GraniteCancellationToken?,
        progressHandler: GraniteOperationProgressHandler?
    ) throws -> PunctuationFormattingResult
}

/// Creates transcript formatter backends from local or Hugging Face checkpoints.
public enum GraniteTranscriptFormatterFactory {
    /// Loads the currently supported punctuation/capitalization architecture.
    ///
    /// Future architectures can be selected here from checkpoint metadata
    /// without changing callers, the CLI, timestamp mapping, or exporters.
    public static func load(
        modelSource: String = PunctuationModelLoader.defaultModelID,
        hfToken: String? = nil,
        progressHandler: GraniteModelDownloadProgressHandler? = nil,
        cancellationToken: GraniteCancellationToken? = nil
    ) throws -> any GraniteTranscriptFormatter {
        try PunctuationFormatter(
            modelSource: modelSource,
            hfToken: hfToken,
            progressHandler: progressHandler,
            cancellationToken: cancellationToken)
    }
}
