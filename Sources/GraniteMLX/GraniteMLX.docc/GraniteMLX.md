# ``GraniteMLX``

Run Granite Speech 5.0 TurboCTC locally in native Swift with MLX or Core ML.

## Overview

GraniteMLX provides media decoding, model acquisition, Granite CTC inference,
approximate word timing, punctuation/capitalization, subtitle segmentation,
and export utilities for Apple Silicon applications.

The high-level pipeline has four independent stages:

1. ``GraniteAudioInput`` decodes a local media file to mono 16 kHz audio.
2. ``GraniteRecognizer`` or ``GraniteCoreMLRecognizer`` produces raw CTC text
   and approximate token/word times.
3. A ``GraniteTranscriptFormatter`` adds non-destructive presentation details.
4. ``GraniteSubtitleSegmenter`` and ``GraniteTranscriptExporter`` create
   display or subtitle output.

Model weights are not bundled with the package. Passing the default Hugging
Face repository IDs downloads and caches the recommended Q8 checkpoints on
first use. Applications can instead pass local checkpoint directories.

### Transcribe a media file

```swift
import Foundation
import GraniteMLX

let cancellation = GraniteCancellationToken()
let audio = try GraniteAudioInput.load(
    url: URL(fileURLWithPath: "/path/to/recording.m4a"),
    cancellationToken: cancellation
)

let recognizer = try GraniteRecognizer(
    modelSource: GraniteModelLoader.defaultModelID,
    cancellationToken: cancellation
)
let raw = try recognizer.transcribe(
    audio,
    activationPrecision: .fp16,
    audioChunkDuration: 122.88,
    audioChunkContext: 10.24,
    cancellationToken: cancellation
)
```

Library calls are synchronous. Run model loading and inference away from an
application's main actor. For long recordings, specify the block-aligned chunk
profile shown above; unlike the CLI, the lower-level recognizer API uses a
single pass when `audioChunkDuration` is omitted.

### Use the published Core ML backend

``GraniteCoreMLRecognizer`` downloads and loads the recommended fixed-shape ML
Program and matching Granite tokenizer/config on first use. It shares audio
preparation, CTC decoding, timestamps, progress, and cancellation behavior
with the MLX path:

```swift
let coreML = try GraniteCoreMLRecognizer(
    modelSource: GraniteCoreMLModelLoader.defaultModelID,
    computeUnits: .cpuAndGPU,
    cancellationToken: cancellation
)
let raw = try coreML.transcribe(
    audio,
    audioChunkContext: 20.48,
    cancellationToken: cancellation
)
```

Omitting the Core ML chunk duration fills the graph after reserving context.
Downloaded repositories use the shared model cache. Compiled packages are
cached under `~/Library/Caches/GraniteMLX/CoreML` by default, and both are
reported and removed together through ``GraniteModelCache``. Applications can
still create a recognizer from local model and tokenizer URLs.

### Apply presentation formatting

Granite's CTC output is generally lowercase and unpunctuated. Load the default
formatter and apply its result without rerunning recognition:

```swift
let formatter = try GraniteTranscriptFormatterFactory.load(
    cancellationToken: cancellation
)
let formatting = try formatter.format(
    raw.rawText,
    cancellationToken: cancellation,
    progressHandler: nil
)
let transcript = raw.applyingFormatting(formatting)
```

Formatting is non-destructive: recognized words and symbols remain
authoritative. A formatter may add capitalization, punctuation, and sentence
boundaries, but unsafe lexical changes fall back to Granite's raw text.

### Create subtitles

Map formatter sentence boundaries onto the recognizer's approximate CTC word
times, then render SRT or WebVTT:

```swift
let segments = GraniteSubtitleSegmenter.segments(
    words: transcript.words,
    sentenceWordRanges: formatting.sentenceWordRanges
)
let srt = GraniteTranscriptExporter.srt(
    segments: segments,
    duration: transcript.duration
)
```

CTC word times are suitable for navigation and ordinary subtitle grouping,
but they are not frame-accurate forced alignment.

### Progress, cancellation, and errors

Create one ``GraniteCancellationToken`` for an application operation and pass
it through loading, recognition, and formatting. Calling
``GraniteCancellationToken/cancel()`` requests cooperative cancellation.
Already-submitted Metal work cannot stop immediately, but cancellation is
checked between stages and chunks.

Audio loading, transcription, and formatting accept a
``GraniteOperationProgressHandler``. Model downloads use the more detailed
``GraniteModelDownloadProgressHandler``. Public operational errors conform to
``GraniteDiagnosticError`` and provide a stable code plus technical context for
logs and support requests.

### Model management

Use ``GraniteModelCatalog`` to resolve published aliases and
``GraniteModelCache`` to inspect, download, or remove compatible checkpoints.
The default cache is the Swift Hugging Face materialization directory. Set
`GRANITE_MLX_HUB_DIRECTORY` before launching a process to isolate that cache.

## Topics

### Recognition

- ``GraniteAudio``
- ``GraniteAudioInput``
- ``GraniteFeatureExtractor``
- ``GraniteRecognizer``
- ``GraniteCoreMLRecognizer``
- ``GraniteCoreMLComputeUnits``
- ``GraniteCoreMLPerformance``
- ``GraniteTranscription``
- ``GraniteWord``
- ``GraniteTokenTiming``
- ``GraniteActivationPrecision``

### Formatting and output

- ``GraniteTranscriptFormatter``
- ``GraniteTranscriptFormatterFactory``
- ``PunctuationFormatter``
- ``PunctuationFormattingResult``
- ``GraniteSubtitleSegmenter``
- ``GraniteSubtitleSegment``
- ``GraniteTranscriptExporter``

### Models and cache

- ``GraniteModelLoader``
- ``GraniteCoreMLModelLoader``
- ``GraniteCoreMLModelArtifact``
- ``GraniteCoreMLModelConfiguration``
- ``GraniteSpeechBackend``
- ``GraniteModelCatalog``
- ``GraniteModelCache``
- ``GranitePublishedModel``
- ``GraniteCachedModel``
- ``GraniteModelDownloadProgress``

### Application integration

- ``GraniteCancellationToken``
- ``GraniteOperationProgress``
- ``GraniteDiagnosticError``
- ``GraniteOperationError``
- ``GraniteAudioError``
- ``GraniteRecognizerError``
- ``GraniteCoreMLRecognizerError``
- ``GraniteModelManagementError``
