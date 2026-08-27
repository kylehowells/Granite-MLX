# Test suites

Run the fast, offline library and CLI contract tests with:

```bash
swift test
```

The offline CLI tests isolate model-cache state with
`GRANITE_MLX_HUB_DIRECTORY`; they never inspect, download, or delete the
user's normal Hugging Face cache.

Run the real-model exporter and media-error integration suite with local
checkpoints and an optional spoken-audio fixture:

```bash
GRANITE_TEST_SPEECH_MODEL=/path/to/granite-q8 \
GRANITE_TEST_PUNCTUATION_MODEL=/path/to/punctuation-q8 \
GRANITE_TEST_AUDIO=/path/to/short-16k-mono.wav \
swift test --filter GraniteMLXCLITests
```

Without `GRANITE_TEST_AUDIO`, the suite creates a deterministic silent WAV.
The local model variables keep release testing offline and prevent CI from
silently downloading more than 500 MB.

The long-form bounded-memory release gate is opt-in:

```bash
GRANITE_TEST_SPEECH_MODEL=/path/to/the-exact-q8-checkpoint \
GRANITE_TEST_LONG_AUDIO=/path/to/long-16k-mono.wav \
GRANITE_TEST_LONG_EXPECTED_TEXT=/path/to/expected-raw-transcript.txt \
GRANITE_TEST_MAX_MLX_PEAK_BYTES=2500000000 \
swift test --filter GraniteMLXCLITests/testOptInLongFormBoundedMemoryRegression
```

The selected expected transcript must come from the exact checkpoint, audio
bytes, activation precision, and chunk profile used by the test. For the
published mixed-G128/G64 Q8 default and the documented CME295 fixture, use
`Benchmarks/bounded-profile-optimization/transcripts/low-context.txt`; the
fixture's SHA-256 is
`f3223fedde7e2212323df3dd59c84193b322fe3c794af60aa9feac7f4044e4ab`.
The gate validates exact raw transcript output, the
122.88-second/10.24-second bounded-memory profile, and MLX peak memory.

Network interruption and resume behavior is also opt-in so ordinary tests do
not depend on Hugging Face availability:

```bash
GRANITE_TEST_NETWORK_MODEL=punctuation-q4 \
swift test --filter GraniteMLXCLITests/testOptInInterruptedDownloadResumesToCompleteCache
```

This test uses an isolated temporary model directory, interrupts the first
download, resumes it, validates the completed cache, and deletes the temporary
directory afterward.

For an entirely local Core ML transcript-parity check, provide the converted
package, its matching tokenizer/config directory, and the known 20-second
audio fixture:

```bash
GRANITE_TEST_COREML_MODEL=/path/to/GraniteSpeech.mlpackage \
GRANITE_TEST_COREML_TOKENIZER=/path/to/granite-tokenizer \
GRANITE_TEST_COREML_AUDIO=/path/to/granite-reference-test-20s.wav \
swift test --filter GraniteMLXCLITests/testOptInCoreMLBackendMatchesReferenceTranscript
```

The published Core ML release lifecycle is a separate opt-in network gate:

```bash
GRANITE_TEST_COREML_NETWORK=1 \
GRANITE_TEST_COREML_AUDIO=/path/to/granite-reference-test-20s.wav \
swift test --filter GraniteMLXCLITests/testOptInPublishedCoreMLFreshDownloadWarmRunAndRemoval
```

It downloads `apache-coreml-q8` into an isolated cache, saves Core ML as the
default in an isolated configuration, verifies the known 20-second transcript,
checks warm reuse and compiled-cache accounting, removes both model caches,
and confirms the model is absent afterward. It intentionally transfers and
compiles about 700 MB, so it does not run as part of the ordinary offline
suite.
