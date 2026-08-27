import Foundation
import Hub
import Tokenizers

final class GraniteTokenizer: @unchecked Sendable {
    private let tokenizer: any Tokenizer

    init(directory: URL) throws {
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        let tokenizerURL = directory.appendingPathComponent("tokenizer.json")
        var object = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any] ?? [:]
        // Granite's ParakeetTokenizer is ordinary byte-level BPE; only its
        // Python class name is custom. Use the equivalent native registration.
        object["tokenizer_class"] = "GPT2Tokenizer"
        let configData = try JSONSerialization.data(withJSONObject: object)
        let tokenizerConfig = try JSONDecoder().decode(Config.self, from: configData)
        let tokenizerData = try JSONDecoder().decode(Config.self, from: Data(contentsOf: tokenizerURL))
        self.tokenizer = try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData,
            strict: true
        )
    }

    func decode(_ tokenIDs: [Int]) -> String {
        tokenizer.decode(tokens: tokenIDs).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func decodeToken(_ tokenID: Int) -> String {
        tokenizer.decode(tokens: [tokenID])
    }

    func tokenString(_ tokenID: Int) -> String? { tokenizer.convertIdToToken(tokenID) }
}
