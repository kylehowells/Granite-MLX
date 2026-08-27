import Foundation

/// An error exposed by GraniteMLX with a stable diagnostic code.
public protocol GraniteDiagnosticError: LocalizedError {
    /// A stable identifier suitable for logs, support requests, and tests.
    var diagnosticCode: String { get }
    /// Low-level context useful when diagnosing the failure.
    var technicalDetails: String? { get }
}

/// A thread-safe, cooperative cancellation token for GraniteMLX operations.
///
/// Cancellation is checked between frontend, model, chunk, formatting, and
/// download stages. Metal work already submitted to the GPU cannot be
/// interrupted, so cancellation of a single unchunked inference is best effort.
public final class GraniteCancellationToken: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    /// Creates an active cancellation token.
    public init() {}

    /// Requests cancellation. Calling this method more than once is harmless.
    public func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }

    /// Indicates whether cancellation has been requested.
    public var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    /// Throws ``GraniteOperationError/cancelled(operation:)`` when cancelled.
    /// - Parameter operation: Human-readable name included in diagnostic details.
    /// - Throws: ``GraniteOperationError/cancelled(operation:)`` after
    ///   ``cancel()`` has been called; otherwise returns normally.
    public func checkCancellation(operation: String) throws {
        if isCancelled { throw GraniteOperationError.cancelled(operation: operation) }
    }
}

/// High-level stages reported by long-running library operations.
public enum GraniteOperationPhase: String, Codable, Sendable {
    /// Input media is being decoded and converted to mono 16 kHz audio.
    case loadingAudio = "loading_audio"
    /// Audio features are being prepared for the speech encoder.
    case extractingFeatures = "extracting_features"
    /// Granite speech inference is running.
    case transcribing
    /// Raw speech text is being punctuated and capitalized.
    case formatting
    /// The requested operation finished successfully.
    case complete
}

/// A progress update emitted by a GraniteMLX operation.
public struct GraniteOperationProgress: Codable, Sendable, Equatable {
    /// Current operation stage.
    public let phase: GraniteOperationPhase
    /// Completion in the inclusive range `0...1`.
    public let fractionCompleted: Double
    /// Human-readable stage description suitable for GUI status text.
    public let message: String
    /// Current zero-based chunk index when chunked inference is active.
    public let chunkIndex: Int?
    /// Total number of chunks when known.
    public let chunkCount: Int?

    /// Creates an operation progress value.
    /// - Parameters:
    ///   - phase: High-level operation stage.
    ///   - fractionCompleted: Completion value; values outside `0...1` are clamped.
    ///   - message: Human-readable status suitable for application UI.
    ///   - chunkIndex: Zero-based active chunk index, or `nil` outside chunked work.
    ///   - chunkCount: Total chunks when known, or `nil` when not applicable.
    public init(
        phase: GraniteOperationPhase,
        fractionCompleted: Double,
        message: String,
        chunkIndex: Int? = nil,
        chunkCount: Int? = nil
    ) {
        self.phase = phase
        self.fractionCompleted = min(1, max(0, fractionCompleted))
        self.message = message
        self.chunkIndex = chunkIndex
        self.chunkCount = chunkCount
    }
}

/// Receives application-facing progress updates from GraniteMLX operations.
public typealias GraniteOperationProgressHandler = @Sendable (GraniteOperationProgress) -> Void

/// Failures shared by cancellable GraniteMLX operations.
public enum GraniteOperationError: Error, GraniteDiagnosticError, Sendable {
    /// Cooperative cancellation was requested by the caller.
    /// - Parameter operation: Human-readable operation interrupted by cancellation.
    case cancelled(operation: String)
    /// An underlying framework produced an error that was not more specifically classified.
    /// - Parameters:
    ///   - code: Stable GraniteMLX diagnostic code for the failed operation.
    ///   - operation: Human-readable operation name.
    ///   - details: Underlying framework error and technical context.
    case underlying(code: String, operation: String, details: String)

    /// Stable diagnostic identifier for the failure.
    public var diagnosticCode: String {
        switch self {
        case .cancelled: "GMLX-OP-001"
        case .underlying(let code, _, _): code
        }
    }

    /// Low-level context useful for diagnostics.
    public var technicalDetails: String? {
        switch self {
        case .cancelled(let operation): "operation=\(operation)"
        case .underlying(_, let operation, let details): "operation=\(operation); underlying=\(details)"
        }
    }

    /// User-facing localized failure description.
    public var errorDescription: String? {
        switch self {
        case .cancelled(let operation):
            "[\(diagnosticCode)] \(operation) was cancelled. Technical details: \(technicalDetails!)."
        case .underlying(_, let operation, _):
            "[\(diagnosticCode)] \(operation) failed. Technical details: \(technicalDetails!)."
        }
    }
}
