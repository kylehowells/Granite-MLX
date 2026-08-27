import Foundation

/// A subtitle cue containing one or more timed words.
public struct GraniteSubtitleSegment: Codable, Sendable, Equatable {
    /// Rendered cue text.
    public let text: String
    /// Cue onset in seconds.
    public let start: Double
    /// Cue end in seconds.
    public let end: Double
    /// Timed words represented by the cue.
    public let words: [GraniteWord]

    /// Creates a subtitle cue.
    /// - Parameters:
    ///   - text: Rendered cue text.
    ///   - start: Cue onset in seconds from the media start.
    ///   - end: Cue end in seconds from the media start.
    ///   - words: Timed words represented by the cue.
    public init(text: String, start: Double, end: Double, words: [GraniteWord]) {
        self.text = text
        self.start = start
        self.end = end
        self.words = words
    }
}

/// Groups timed words into readable subtitle cues.
public enum GraniteSubtitleSegmenter {
    /// Segments words at formatter boundaries, pauses, and configured limits.
    /// - Parameters:
    ///   - words: Chronological timed words to group into cues.
    ///   - sentenceWordRanges: Formatter-provided half-open sentence ranges into
    ///     `words`; sentence ends become preferred hard boundaries.
    ///   - maxWords: Maximum words per cue before starting another cue.
    ///   - silenceGap: Word-onset gap in seconds that starts a new cue.
    ///   - maxDuration: Maximum onset span in seconds within one cue.
    /// - Returns: Chronological subtitle cues. Empty input produces no cues.
    public static func segments(
        words: [GraniteWord],
        sentenceWordRanges: [Range<Int>] = [],
        maxWords: Int = 20,
        silenceGap: Double = 1,
        maxDuration: Double = 8
    ) -> [GraniteSubtitleSegment] {
        guard !words.isEmpty else { return [] }
        let boundaries = Set(sentenceWordRanges.dropLast().map(\.upperBound))
        var result: [GraniteSubtitleSegment] = []
        var current: [GraniteWord] = []

        func finish() {
            guard let first = current.first, let last = current.last else { return }
            result.append(GraniteSubtitleSegment(
                text: current.map(\.text).joined(separator: " "),
                start: first.start,
                end: max(first.start, last.end),
                words: current
            ))
            current.removeAll(keepingCapacity: true)
        }

        for (index, word) in words.enumerated() {
            if let first = current.first, let previous = current.last {
                let shouldSplit = boundaries.contains(index)
                    || word.start - previous.start >= silenceGap
                    || current.count >= maxWords
                    || word.start - first.start >= maxDuration
                if shouldSplit { finish() }
            }
            current.append(word)
        }
        finish()
        return result
    }
}

/// Renders transcription output in plain-text and subtitle formats.
public enum GraniteTranscriptExporter {
    /// Renders user-facing transcript text with a trailing newline.
    /// - Parameter transcription: Raw or formatted transcription to render.
    /// - Returns: ``GraniteTranscription/text`` followed by one newline.
    public static func text(_ transcription: GraniteTranscription) -> String {
        transcription.text + "\n"
    }

    /// Renders SubRip subtitle cues.
    /// - Parameters:
    ///   - segments: Chronological subtitle cues.
    ///   - duration: Media duration used for the no-speech fallback cue.
    ///   - highlightWords: When `true`, emits one cue per active word and wraps
    ///     that word in SubRip-compatible underline tags.
    /// - Returns: Complete UTF-8 SubRip text with comma millisecond separators.
    public static func srt(
        segments: [GraniteSubtitleSegment],
        duration: Double,
        highlightWords: Bool = false
    ) -> String {
        subtitle(
            segments: segments, duration: duration, highlightWords: highlightWords,
            isWebVTT: false)
    }

    /// Renders WebVTT subtitle cues.
    /// - Parameters:
    ///   - segments: Chronological subtitle cues.
    ///   - duration: Media duration used for the no-speech fallback cue.
    ///   - highlightWords: When `true`, emits one cue per active word and wraps
    ///     that word in WebVTT bold tags.
    /// - Returns: Complete WebVTT text beginning with the `WEBVTT` header.
    public static func webVTT(
        segments: [GraniteSubtitleSegment],
        duration: Double,
        highlightWords: Bool = false
    ) -> String {
        "WEBVTT\n\n" + subtitle(
            segments: segments, duration: duration, highlightWords: highlightWords,
            isWebVTT: true)
    }

    private static func subtitle(
        segments: [GraniteSubtitleSegment],
        duration: Double,
        highlightWords: Bool,
        isWebVTT: Bool
    ) -> String {
        let cues: [(Double, Double, String)]
        if segments.isEmpty {
            cues = [(0, max(0, duration), "[no speech detected]")]
        } else if highlightWords {
            cues = segments.flatMap { segment in
                segment.words.indices.map { activeIndex in
                    let active = segment.words[activeIndex]
                    let rendered = segment.words.enumerated().map { index, word in
                        let escaped = escapeSubtitle(word.text)
                        guard index == activeIndex else { return escaped }
                        return isWebVTT ? "<b>\(escaped)</b>" : "<u>\(escaped)</u>"
                    }.joined(separator: " ")
                    return (active.start, max(active.start, active.end), rendered)
                }
            }
        } else {
            cues = segments.map { ($0.start, $0.end, escapeSubtitle($0.text)) }
        }

        return cues.enumerated().map { index, cue in
            let separator: Character = isWebVTT ? "." : ","
            let identifier = isWebVTT ? "" : "\(index + 1)\n"
            return "\(identifier)\(timestamp(cue.0, separator: separator)) --> \(timestamp(cue.1, separator: separator))\n\(cue.2)\n"
        }.joined(separator: "\n") + "\n"
    }

    private static func timestamp(_ seconds: Double, separator: Character) -> String {
        let totalMilliseconds = max(0, Int((seconds * 1_000).rounded()))
        let milliseconds = totalMilliseconds % 1_000
        let totalSeconds = totalMilliseconds / 1_000
        let second = totalSeconds % 60
        let totalMinutes = totalSeconds / 60
        let minute = totalMinutes % 60
        let hour = totalMinutes / 60
        return String(
            format: "%02d:%02d:%02d%@%03d",
            hour, minute, second, String(separator), milliseconds)
    }

    private static func escapeSubtitle(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }
}
