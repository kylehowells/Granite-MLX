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
    /// - Parameters:
    ///   - architecture: Formatter architecture identifier used for backend selection.
    ///   - precision: Human-readable checkpoint precision such as `Q8` or `FP16`.
    ///   - quantizationBits: Packed weight bit width, or `nil` for floating weights.
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
    /// - Parameters:
    ///   - text: Raw lexical transcript to annotate without replacing its words.
    ///   - cancellationToken: Optional cooperative cancellation token.
    ///   - progressHandler: Receives formatter window and completion progress.
    /// - Returns: Presentation text and sentence-to-word boundary mapping.
    /// - Throws: ``GraniteOperationError`` when cancelled or an underlying
    ///   operation fails, and backend-specific validation errors.
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
    /// - Parameters:
    ///   - modelSource: Local checkpoint directory, catalog alias, or Hugging
    ///     Face repository ID. The recommended Q8 formatter is used when omitted.
    ///   - storage: Explicit model and transfer-cache locations.
    ///   - hfToken: Optional Hugging Face token for private or gated repositories.
    ///   - progressHandler: Receives model cache and download events.
    ///   - cancellationToken: Cooperatively cancels model acquisition.
    /// - Returns: A loaded formatter behind the architecture-independent protocol.
    /// - Throws: Model-management, checkpoint-validation, tokenizer-loading, or
    ///   cancellation errors produced while constructing the formatter.
    public static func load(
        modelSource: String = PunctuationModelLoader.defaultModelID,
        storage: GraniteModelStorage = .default,
        hfToken: String? = nil,
        progressHandler: GraniteModelDownloadProgressHandler? = nil,
        cancellationToken: GraniteCancellationToken? = nil
    ) throws -> any GraniteTranscriptFormatter {
        try PunctuationFormatter(
            modelSource: modelSource,
            storage: storage,
            hfToken: hfToken,
            progressHandler: progressHandler,
            cancellationToken: cancellationToken)
    }
}
