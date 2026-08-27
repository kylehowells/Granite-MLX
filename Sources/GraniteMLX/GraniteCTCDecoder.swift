public struct GraniteTokenTiming: Codable, Sendable, Equatable {
    public let tokenID: Int
    public let text: String
    public let start: Double
    public let end: Double

    public init(tokenID: Int, text: String, start: Double, end: Double) {
        self.tokenID = tokenID
        self.text = text
        self.start = start
        self.end = end
    }
}

public enum GraniteCTCDecoder {
    /// Greedy CTC collapse: merge adjacent repeats, then remove blank ID 0.
    public static func collapse(_ frameTokenIDs: [Int], blankID: Int = 0) -> [Int] {
        var result: [Int] = []
        result.reserveCapacity(frameTokenIDs.count)
        var previous = -1
        for token in frameTokenIDs {
            if token != previous && token != blankID { result.append(token) }
            previous = token
        }
        return result
    }

    /// Converts greedy CTC frame predictions into timestamped token emissions.
    /// Repeated adjacent IDs are one emission; blank frames terminate it.
    public static func tokenTimings(
        _ frameTokenIDs: [Int],
        frameRate: Double,
        blankID: Int = 0,
        decodeToken: (Int) -> String
    ) -> [GraniteTokenTiming] {
        guard frameRate > 0 else { return [] }
        var result: [GraniteTokenTiming] = []
        var activeID: Int?
        var activeStart = 0
        var activeEnd = 0

        func finishActive() {
            guard let tokenID = activeID else { return }
            result.append(GraniteTokenTiming(
                tokenID: tokenID,
                text: decodeToken(tokenID),
                start: Double(activeStart) / frameRate,
                end: Double(activeEnd + 1) / frameRate
            ))
            activeID = nil
        }

        for (frame, tokenID) in frameTokenIDs.enumerated() {
            if tokenID == blankID {
                finishActive()
            } else if tokenID == activeID {
                activeEnd = frame
            } else {
                finishActive()
                activeID = tokenID
                activeStart = frame
                activeEnd = frame
            }
        }
        finishActive()
        return result
    }

    /// Groups decoded token pieces into words while retaining their CTC times.
    public static func words(from tokens: [GraniteTokenTiming]) -> [GraniteWord] {
        var words: [GraniteWord] = []
        var text = ""
        var start: Double?
        var end = 0.0

        func finishWord() {
            guard !text.isEmpty, let wordStart = start else { return }
            words.append(GraniteWord(text: text, start: wordStart, end: end))
            text = ""
            start = nil
        }

        for token in tokens {
            for character in token.text {
                if character.isWhitespace {
                    finishWord()
                } else {
                    if start == nil { start = token.start }
                    text.append(character)
                    end = token.end
                }
            }
        }
        finishWord()

        // Match the reference implementation: keep a spoken word displayed
        // until the next word starts, unless that gap is implausibly long.
        guard words.count > 1 else { return words }
        return words.enumerated().map { index, word in
            guard index + 1 < words.count else { return word }
            let nextStart = words[index + 1].start
            return GraniteWord(
                text: word.text,
                start: word.start,
                end: nextStart - word.start <= 2 ? nextStart : word.end)
        }
    }
}
