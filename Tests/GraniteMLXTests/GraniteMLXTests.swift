import XCTest
@testable import GraniteMLX

final class GraniteMLXTests: XCTestCase {
    func testLocalPunctuationFormatterMatchesMLXReference() throws {
        guard let path = ProcessInfo.processInfo.environment["GRANITE_PUNCTUATION_MODEL"] else {
            throw XCTSkip("Set GRANITE_PUNCTUATION_MODEL to run the local formatter parity test.")
        }
        let formatter = try PunctuationFormatter(modelURL: URL(fileURLWithPath: path))
        let result = formatter.format(
            "hello hello everyone and welcome to cme 295 transformers and large language models")
        if path.hasSuffix("mlx-q5") {
            XCTAssertEqual(
                result.text,
                "Hello, hello everyone? And welcome to CME 295 transformers and large language models.")
            XCTAssertEqual(result.sentences.count, 2)
        } else {
            XCTAssertEqual(
                result.text,
                "Hello, hello everyone, and welcome to CME 295 Transformers and large language models.")
            XCTAssertEqual(result.sentences.count, 1)
        }
    }

    func testFullLocalPunctuationFormatterMatchesPythonQ8() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["GRANITE_PUNCTUATION_MODEL"],
              let inputPath = environment["GRANITE_PUNCTUATION_INPUT"],
              let referencePath = environment["GRANITE_PUNCTUATION_REFERENCE"] else {
            throw XCTSkip("Set the three GRANITE_PUNCTUATION_* paths for full parity.")
        }
        let formatter = try PunctuationFormatter(modelURL: URL(fileURLWithPath: modelPath))
        let input = try String(contentsOfFile: inputPath, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let expected = try String(contentsOfFile: referencePath, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertEqual(formatter.format(input).text, expected)
    }

    func testAudioDuration() {
        let audio = GraniteAudio(samples: [0, 0, 0, 0], sampleRate: 2, source: URL(fileURLWithPath: "/tmp/test.wav"))
        XCTAssertEqual(audio.duration, 2.0)
    }

    func testGreedyCTCCollapse() {
        XCTAssertEqual(
            GraniteCTCDecoder.collapse([0, 4, 4, 0, 4, 7, 7, 0]),
            [4, 4, 7]
        )
    }

    func testCTCWordTimingsFromDecodedPieces() {
        let pieces = [1: " hello", 2: " wor", 3: "ld"]
        let tokens = GraniteCTCDecoder.tokenTimings(
            [0, 1, 1, 0, 2, 2, 3, 0],
            frameRate: 10,
            decodeToken: { pieces[$0]! })
        XCTAssertEqual(tokens.map(\.tokenID), [1, 2, 3])
        XCTAssertEqual(tokens[0].start, 0.1, accuracy: 0.0001)
        XCTAssertEqual(tokens[0].end, 0.3, accuracy: 0.0001)

        let words = GraniteCTCDecoder.words(from: tokens)
        XCTAssertEqual(words.map(\.text), ["hello", "world"])
        XCTAssertEqual(words[0].start, 0.1, accuracy: 0.0001)
        XCTAssertEqual(words[0].end, 0.4, accuracy: 0.0001)
        XCTAssertEqual(words[1].start, 0.4, accuracy: 0.0001)
        XCTAssertEqual(words[1].end, 0.7, accuracy: 0.0001)
    }

    func testSubtitleSegmentationUsesSentenceBoundariesAndLimits() {
        let words = [
            GraniteWord(text: "Hello.", start: 0, end: 0.5),
            GraniteWord(text: "This", start: 0.5, end: 0.8),
            GraniteWord(text: "works.", start: 0.8, end: 1.2),
        ]
        let segments = GraniteSubtitleSegmenter.segments(
            words: words,
            sentenceWordRanges: [0 ..< 1, 1 ..< 3])
        XCTAssertEqual(segments.map(\.text), ["Hello.", "This works."])
        XCTAssertEqual(segments[1].start, 0.5, accuracy: 0.0001)
        XCTAssertEqual(segments[1].end, 1.2, accuracy: 0.0001)
    }

    func testSubtitleExportersAndWordHighlighting() {
        let words = [
            GraniteWord(text: "Hello", start: 1, end: 1.5),
            GraniteWord(text: "world.", start: 1.5, end: 2),
        ]
        let segment = GraniteSubtitleSegment(
            text: "Hello world.", start: 1, end: 2, words: words)
        XCTAssertEqual(
            GraniteTranscriptExporter.srt(segments: [segment], duration: 2),
            "1\n00:00:01,000 --> 00:00:02,000\nHello world.\n\n")
        let highlighted = GraniteTranscriptExporter.webVTT(
            segments: [segment], duration: 2, highlightWords: true)
        XCTAssertTrue(highlighted.hasPrefix("WEBVTT\n\n"))
        XCTAssertTrue(highlighted.contains("<b>Hello</b> world."))
        XCTAssertTrue(highlighted.contains("Hello <b>world.</b>"))
    }

    func testPublishedModelCatalogAliasesAndDefaults() throws {
        XCTAssertEqual(GraniteModelCatalog.models.count, 15)
        XCTAssertEqual(Set(GraniteModelCatalog.models.map(\.alias)).count, 15)
        XCTAssertEqual(Set(GraniteModelCatalog.models.map(\.repositoryID)).count, 15)
        XCTAssertEqual(GraniteModelCatalog.models.filter(\.isDefault).count, 2)
        let resolved = try GraniteModelCatalog.resolve("apache-q8")
        XCTAssertEqual(resolved.id, GraniteModelLoader.defaultModelID)
        XCTAssertEqual(resolved.model?.kind, .speech)
        XCTAssertThrowsError(try GraniteModelCatalog.resolve("not-a-model"))
    }

    func testModelCacheDirectorySizeDoesNotFollowSymlinks() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(repeating: 1, count: 1_024).write(to: directory.appendingPathComponent("one.bin"))
        let nested = directory.appendingPathComponent("nested", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try Data(repeating: 2, count: 2_048).write(to: nested.appendingPathComponent("two.bin"))
        try FileManager.default.createSymbolicLink(
            at: directory.appendingPathComponent("link.bin"),
            withDestinationURL: nested.appendingPathComponent("two.bin"))
        XCTAssertEqual(GraniteModelCache.directorySize(directory), 3_072)
    }

    func testModelCacheDetectionRejectsUnrelatedHubModels() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data().write(to: directory.appendingPathComponent("model.safetensors"))
        try Data("{}".utf8).write(to: directory.appendingPathComponent("tokenizer.json"))
        try Data(#"{"model_type":"moonshine_streaming"}"#.utf8)
            .write(to: directory.appendingPathComponent("config.json"))
        XCTAssertNil(GraniteModelCache.detectedKind(at: directory))

        try Data(#"{"model_type":"granite_speech5_ctc"}"#.utf8)
            .write(to: directory.appendingPathComponent("config.json"))
        XCTAssertEqual(GraniteModelCache.detectedKind(at: directory), .speech)

        try Data(#"{"architecture":"bert-punctuation-capitalization-segmentation"}"#.utf8)
            .write(to: directory.appendingPathComponent("mlx_config.json"))
        XCTAssertEqual(GraniteModelCache.detectedKind(at: directory), .punctuation)
    }

    func testQuantizationConfigurationDecodes() throws {
        let data = Data("""
        {
          "model_type": "granite_speech5_ctc",
          "vocab_size": 16384,
          "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
        }
        """.utf8)
        let configuration = try JSONDecoder().decode(GraniteModelConfiguration.self, from: data)
        XCTAssertEqual(
            configuration.quantization,
            GraniteQuantizationConfiguration(groupSize: 64, bits: 4, mode: .affine)
        )
    }

    func testMixedQuantizationGroupSizesDecodeAndResolve() throws {
        let data = Data("""
        {
          "model_type": "granite_speech5_ctc",
          "vocab_size": 16384,
          "quantization": {
            "group_size": 128,
            "bits": 8,
            "mode": "affine",
            "group_sizes": {"encoder.input_linear": 64}
          }
        }
        """.utf8)
        let model = try JSONDecoder().decode(GraniteModelConfiguration.self, from: data)
        let quantization = try XCTUnwrap(model.quantization)
        XCTAssertEqual(quantization.groupSize(for: "encoder.input_linear"), 64)
        XCTAssertEqual(quantization.groupSize(for: "encoder.layers.0.self_attn.linear_q"), 128)
    }

    func testMixedQuantizedWeightValidationUsesModuleOverride() throws {
        let shapes = [
            "encoder.input_linear.weight": [4, 16],
            "encoder.input_linear.scales": [4, 1],
            "encoder.input_linear.biases": [4, 1],
            "encoder.output.weight": [4, 32],
            "encoder.output.scales": [4, 1],
            "encoder.output.biases": [4, 1],
        ]
        try GraniteModelLoader.validateQuantizedWeightShapes(
            shapes,
            configuration: GraniteQuantizationConfiguration(
                groupSize: 128,
                bits: 8,
                mode: .affine,
                groupSizes: ["encoder.input_linear": 64]
            )
        )
    }

    func testQuantizedWeightValidationAcceptsConsistentAffineWeights() throws {
        let shapes = [
            "layer.weight": [4, 16],
            "layer.scales": [4, 1],
            "layer.biases": [4, 1],
        ]
        try GraniteModelLoader.validateQuantizedWeightShapes(
            shapes,
            configuration: GraniteQuantizationConfiguration(
                groupSize: 64, bits: 8, mode: .affine)
        )
    }

    func testQuantizedWeightValidationRejectsMissingScales() {
        let shapes = [
            "layer.weight": [4, 16],
        ]
        XCTAssertThrowsError(
            try GraniteModelLoader.validateQuantizedWeightShapes(
                shapes,
                configuration: GraniteQuantizationConfiguration(
                    groupSize: 64, bits: 8, mode: .affine)
            )
        )
    }

}
