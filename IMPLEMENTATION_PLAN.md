# Granite-MLX Implementation Plan

Native Swift/MLX command-line transcription for IBM Granite Speech 5.0 TurboCTC on Apple Silicon.

## Goal

Build a distributable macOS command-line tool named `granite-mlx` that runs an MLX-converted version of:

`ibm-granite/granite-speech-5.0-470m-turboctc`

Granite Speech 5.0 is available in multiple weight-license variants,
including an Apache-licensed variant and a non-commercial variant. The
Granite-MLX software must support either compatible checkpoint; the user
selects the model repository or local model directory appropriate for their
use. The software license and model-weight license remain separate.

The project should provide a native Swift runtime, a Hugging Face/local model loader, practical audio transcription output, reproducible conversion and validation scripts, and a release path suitable for GitHub Releases and a Homebrew tap.

The original model is an English-only, encoder-only CTC model. It uses a 16-layer Conformer encoder, 1,024 hidden dimensions, a 16,384-token BPE vocabulary, 16 kHz audio, 80-bin log-mel features plus deltas, temporal subsampling by 8, block attention, and greedy CTC decoding.

Model weights remain outside this repository. The authoritative Apache source
checkpoint is
[`ibm-granite/granite-speech-5.0-470m-turboctc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc),
and users can select its compatible non-commercial counterpart when those
weight terms are appropriate. Converted MLX checkpoints are published in
separate Hugging Face repositories and downloaded and cached by the CLI.

## Active remaining work (2026-08-28)

This section is the authoritative checklist for the first public release. The
phase sections below preserve implementation history and longer-term ideas;
unchecked post-release items in those sections do not block the first public
release.

### Final release verification

- [x] Run a final short transcription from the current commit with the default
  MLX Q8 speech and punctuation models and inspect the formatted output. The raw
  transcript matches the fixed reference and formatting is sensible for the
  deliberately mid-sentence 20-second cut.
- [x] Run the short Core ML parity fixture from the current commit. The direct
  FP16 input-copy path produces the exact fixed reference transcript.
- [x] Create and push the first semantic-version tag. `0.1.0` validated the
  release machinery; `0.1.1` supersedes it with corrected installation
  documentation and is the release intended for Homebrew.
- [x] Confirm the `0.1.1` release workflow publishes the macOS ARM64 archive and
  its SHA-256 checksum without bundling model weights.
- [x] Download the public `0.1.1` archive independently, verify its checksum and
  contents, and run the downloaded binary with an empty `PATH`. The complete
  release lifecycle was exercised with isolated model/config/cache directories:
  `--version`, `--help`, model listing, first-run Q8 speech and punctuation
  downloads, cache reuse, formatted transcription, and model removal. A separate
  physical clean-Mac test remains part of Homebrew formula validation.

### Homebrew distribution

- [ ] Create or update `kylehowells/homebrew-tap`.
- [ ] Add a binary `Formula/granite-mlx.rb` referencing the `0.1.1` archive and
  checksum, declaring Apple Silicon, minimum macOS, and `ffmpeg` requirements.
- [ ] Add model-free formula tests for `--version`, `--help`, and `models list`.
- [ ] Test tap installation, upgrade, uninstall, reinstall, and first-run model
  download/transcription on a clean Mac.

### Post-release engineering (not `0.1.0` blockers)

- [ ] Add portable frontend/intermediate-tensor numerical fixtures and broader
  frontend unit coverage.
- [ ] Stream audio decoding/resampling into inference windows to reduce long-file
  memory, and implement exact layer-wise streaming if exact one-pass parity is
  eventually required.
- [ ] Validate MLX Q8/Q6 and Core ML on physical iPhone/iPad hardware and newer
  Apple Silicon generations.
- [ ] Obtain independently verified transcripts or a curated evaluation corpus
  before publishing WER claims.
- [ ] Pursue the separate post-`0.1.0` Linux/CUDA plan in `LINUX_SUPPORT.md`.

## Reference projects

- `moonshine-mlx` — native Swift speech-recognition runtime and CLI.
- `DeepFilterNet-mlx` — MLX conversion, audio processing, benchmarking, and packaging patterns.
- `demucs-mlx-swift` — Swift MLX audio model loading and CLI structure.
- `Moebius-MLX` — larger custom MLX model architecture and performance work.
- `senstella/parakeet-mlx` — behavioral reference for the existing Python Parakeet CLI.
- `Blaizzy/mlx-audio-swift` — existing Swift MLX audio model conventions, including older Granite Speech support.

## Phase 1: Repository foundation

- [x] Add `Package.swift` using Swift 6.x, macOS 14+, and Apple Silicon as the initial target.
- [x] Depend on `mlx-swift`, `swift-transformers`, and `swift-argument-parser`.
- [x] Create library target `GraniteMLX`.
- [x] Create executable target `granite-mlx`.
- [x] Add test target and initial test fixtures.
- [x] Pin known-good MLX Swift 0.31.4, Swift Transformers 1.3.3, and Swift
  Argument Parser 1.8.2 releases and document the minimum Xcode/Swift/macOS versions.
- [x] Treat `mlx-swift-lm` as a reference for package/model-loading
  conventions while implementing Granite as a custom encoder rather than an
  LLM registry entry.
- [x] Add `.gitignore` for `.build`, model directories, caches, generated outputs, and local benchmark artifacts.
- [x] Dual-license the software under Apache-2.0 or MIT, at the user's option,
  independently of all model-weight licenses.

## Phase 2: PyTorch reference runner

- [x] Build a small Python reference runner using the downloaded Transformers model.
- [x] Define the CLI contract and output formats in `Scripts/granite_reference.py`.
- [x] Match Parakeet-style `txt`, `srt`, `vtt`, `json`, `all`, and highlighted word-cue exports in the Python reference.
- [x] Validate model loading and JSON output against the local Granite checkpoint.
- [x] Load audio at 16 kHz mono and run the official processor/model path.
- [x] Verify manual greedy CTC decoding produces the same token IDs and transcript as the official `model.generate()` path.
- [-] Save deterministic intermediate artifacts for a short fixture. Final
  transcript/token parity is complete; frontend features, encoder outputs,
  middle self-conditioned output, and logits remain an optional post-`0.1.0`
  diagnostic fixture.
- [x] Record the exact Transformers version and source-model revision in the
  checked benchmark metadata, conversion manifests, and published model cards.

## Phase 3: MLX conversion

- [x] Add `Scripts/convert_granite.py`.
- [x] Accept a source model ID/path as an argument so either Granite 5.0 weight-license variant can be converted.
- [x] Convert the original safetensors checkpoint into MLX-loadable safetensors.
- [x] Preserve or generate the required `config.json`, tokenizer files, processor metadata, and model manifest.
- [x] Remove training-only state such as BatchNorm `num_batches_tracked` values.
- [x] Verify every required tensor name and shape against the Swift module tree
  and reject missing, extra training-only, or inconsistent quantized tensors.
- [x] Document all layout changes, especially linear and convolution weight transpositions.
- [x] Produce at least:
  - [x] BF16 or FP16 baseline checkpoint
  - [x] FP32 validation checkpoint if needed
  - [x] affine Q8/Q6/Q5/Q4/Q3/Q2 checkpoints after numerical parity was established
- [x] Store quantization mode, bit width, and group size in `config.json`.
- [x] Validate packed weight, scale, and quantization-bias shapes at load time.
- [x] Document source models, output formats, conversion commands, expected
  sizes, and manifests in the project README and published checkpoint cards.

## Phase 4: MLX numerical parity

- [-] Implement an MLX Python model matching the Granite 5.0 architecture.
  Superseded by the native Swift/MLX implementation and its established output
  parity; a second complete MLX runtime is not required for `0.1.0`.
- [x] Match the frontend exactly:
  - [x] 16 kHz input
  - [x] 80 mel bins
  - [x] log-mel floor and normalization
  - [x] delta features
  - [x] frame stacking
  - [x] padding and length handling
- [x] Match the encoder exactly in native Swift:
  - [x] input projection
  - [x] relative positional embeddings
  - [x] block-wise attention with block size 128
  - [x] depthwise convolution module
  - [x] first two block subsampling behavior
  - [x] self-conditioned middle CTC path
  - [x] final CTC projection
- [-] Add a dedicated PyTorch-to-MLX intermediate comparison script. Existing
  conversion, frontend-dump, transcript-parity, and quantization tooling cover
  the release gates; intermediate tensor comparison remains optional follow-up.
- [-] Compare PyTorch and MLX intermediate tensors with tolerances appropriate
  to BF16/FP16 as part of that optional diagnostic work.
- [x] Compare final decoded text with the Python reference on the fixed 20-second fixture.
- [x] Treat transcript parity as a release gate.

## Phase 5: Native Swift runtime

- [x] Add `GraniteConfiguration` decoded from `config.json`.
- [x] Add `GraniteModelLoader` supporting:
  - [x] local model directories
  - [x] Hugging Face repository IDs
  - [x] model caching
  - [x] download progress
  - [x] optional Hugging Face authentication through `--hf-token` or `HF_TOKEN`
- [x] Add `GraniteAudioProcessor` using MLX operations for the feature frontend, including torchaudio-compatible reflect padding.
- [x] Add native Swift audio loading and resampling to 16 kHz mono.
- [x] Add ffmpeg fallback for video and audio containers that AVFoundation cannot decode directly.
- [x] Continue a multi-input transcription batch after per-file failures, retain
  successful outputs, and return failure after reporting the batch summary.
- [x] Add structured, coded library errors plus application-facing cooperative
  cancellation and progress callbacks for audio, model download, inference,
  and punctuation formatting.
- [x] Add `///` API documentation to the exported library surface.
- [x] Add Conformer modules and weight loading with explicit key mapping.
- [x] Add block attention and relative-position handling.
- [x] Add strided convolution and residual pooling for temporal subsampling.
- [x] Add self-conditioned CTC computation.
- [x] Add greedy CTC collapse with blank ID 0.
- [x] Add BPE tokenizer decoding through `swift-transformers`.
- [x] Add approximate word alignment from CTC emission frames, using next-word onset for non-final word cue ends.
- [x] Add subtitle segmentation by onset gaps, maximum words, and maximum duration.
- [x] Add timestamp-aware central-emission cropping for overlapping chunks,
  followed by one global CTC collapse and tokenizer decode; no text-only merge
  is used.
- [x] Add converted FP16 support and explicit synchronization/evaluation points.
- [x] Add Swift model artifact loading and required-tensor audit.
- [x] Add native affine quantized Linear/Embedding loading for Q8 through Q2.
- [x] Add mixed per-module quantization group sizes and convert the recommended
  Q8 checkpoint with G128 weights plus the required G64 input projection.
- [x] Add FP16 encoder activations with FP32 softmax stability islands and make
  FP16 the optimized CLI default.
- [x] Add explicit FP8-E4M3 and signed-INT8 emulated activation-storage modes
  for capability and accuracy experiments.
- [x] Add activation dtype/range auditing.

## Phase 6: CLI behavior

Initial command:

```bash
granite-mlx audio.wav
```

Required options:

- [x] Multiple input audio paths.
- [x] `--model` for local directories and Hugging Face IDs.
- [x] Support selecting either compatible Granite 5.0 model variant through `--model` without hard-coding a license-specific checkpoint.
- [x] `--output-dir`.
- [x] `--output-format txt|srt|vtt|json|all`.
- [x] `--output-template`.
- [x] `--audio-chunk-duration` and `--audio-chunk-context` for long audio,
  plus `--no-chunking` and `--audio-chunk-duration 0` one-pass aliases.
- [x] `--highlight-words` for word-highlighted SRT/VTT.
- [x] `--max-words`, `--silence-gap`, and `--max-duration` for subtitle segmentation.
- [x] Select weight precision through `--model` and activation precision through
  `--activation-precision`; dedicated ambiguous `--bf16`/`--fp32` flags are not
  part of the final CLI contract.
- [x] `--verbose`.
- [x] `--version` (provisional `0.1.0`; derive it from release metadata before tagging).
- [x] `--help` generated by ArgumentParser (more examples remain for final CLI polish).
- [x] Add persistent `granite-mlx config show|get|set|unset` commands. Store the
  selected default backend as versioned JSON under Application Support, apply
  precedence `--backend` → saved setting → built-in MLX, and avoid requiring a
  shell environment variable.

Output behavior:

- [x] Print the selected SRT/TXT/VTT/JSON output to stdout when no output directory is supplied.
- [x] Write formatted files when `--output-dir` or an output format is requested.
- [x] Keep machine-readable JSON free of progress text on stdout.
- [x] Send diagnostics and benchmark records to stderr.
- [x] Include model, duration, load time, transcription time, real-time factor, and precision in JSON.
- [x] Match the Python reference's default SRT behavior and output-template variables (`filename`, `parent`, `index`, `date`).

Word timestamps are not directly produced by the model. Swift and Python now
derive approximate word timings from CTC emission frames and export
Parakeet-style timestamped SRT/VTT. These are subtitle-grade estimates rather
than forced alignment.

### Punctuation, capitalization, and sentence boundaries

Granite's CTC transcript is lowercase and generally unpunctuated, whereas
Parakeet emits presentation-ready casing and punctuation directly. Match that
user-facing behavior with the independently licensed English punctuation,
capitalization, and segmentation model derived from
`1-800-BAD-CODE/punctuation_fullstop_truecase_english`.

- [x] Run the complete 101m59s Granite Q8 transcript through the original
  ONNX FP32 implementation and save its formatted baseline and timing data.
- [x] Add a reproducible ONNX-to-MLX converter for this BERT-based formatter.
- [x] Produce and validate MLX FP16, Q8, Q6, Q5, and Q4 formatter checkpoints.
- [x] Measure checkpoint size, speed, RSS, sentence count, and character/word
  disagreement against ONNX FP32; save JSON and PNG results.
- [x] Select Q8/G64 as the recommended formatter checkpoint: 56.4 MB total,
  0.26s model inference for the full lecture transcript, and 99.954% character
  agreement with ONNX FP32 in the current benchmark.
- [x] Re-run the complete lecture through Parakeet MLX and Granite Q8 plus MLX
  Q8 formatting, preserving both formatted transcripts and timing records.
- [x] Port the validated MLX formatter architecture and tokenizer to Swift;
  native Q8 output exactly matches the Python MLX Q8 output on the complete
  101m59s transcript.
- [x] Publish each formatter precision as a separate Hugging Face repository;
  auto-download/cache the recommended Q8 checkpoint independently of Granite.
- [x] Add `--punctuate` and `--no-punctuate`; make formatted output the default
  while keeping raw Granite output available for parity tests and constrained
  environments.
- [x] Preserve raw and formatted text in the library result type so clients do
  not need to rerun ASR to switch presentation modes.
- [x] Return formatter sentence-boundary word indices and map them onto
  Granite's CTC word timestamps. Use those boundaries for SRT/VTT grouping,
  with silence gap, maximum duration, and maximum words as safety limits.
- [x] Add deterministic Swift-vs-Python-MLX formatter parity fixtures: all five
  variants match their short Python reference output, and Q8 matches the full
  101m59s transcript character-for-character.
- [x] Include formatter load/inference time and model ID in CLI benchmark JSON.
- [x] Include formatter precision and both raw/formatted transcript fields in
  the full transcription JSON exporter.
- [x] Make formatting non-destructive: align the formatter output to Granite's
  original words, restore tokenizer-unknown symbols such as `%` and hyphens,
  reject lexical substitutions, and fall back to raw text when alignment is
  unsafe.
- [x] Put formatter loading behind `GraniteTranscriptFormatter` and
  `GraniteTranscriptFormatterFactory` so a future formatter architecture can
  replace the current BERT model without changing the recognizer, CLI,
  timestamp mapping, or exporters.
- [x] Add regression coverage for unknown-symbol restoration, lexical-change
  rejection, and unsafe word-count alignment. Verify the four extracted
  problem clips and the complete 101m59s transcript contain no `<unk>` output
  and preserve a one-to-one mapping to Granite's raw words.

## Phase 7: Tests and benchmarks

- [ ] Add frontend unit tests for shape, sample rate, normalization, deltas, and padding.
- [x] Add CTC decoder tests for blanks and repeated tokens.
- [x] Add test-only configuration, quantized-weight, model-cache, Core ML
  artifact, and published-model loading coverage, with full checkpoints behind
  opt-in integration gates.
- [-] Add a portable checked-in PyTorch-vs-MLX tensor fixture. End-to-end fixed
  audio parity and saved transcripts are complete; intermediate tensor parity
  remains an optional post-`0.1.0` diagnostic.
- [x] Validate Swift end-to-end transcription externally against the fixed Python fixture (a portable checked-in fixture test remains).
- [x] Add benchmark mode reporting load time, inference time, RTF, and throughput (peak memory remains external via `/usr/bin/time -lp`).
- [x] Add a reproducible Q8-to-Q2 benchmark harness covering file size, RSS,
  macOS peak footprint, speed, and transcript disagreement.
- [x] Save quantization benchmark runs and summary metrics as JSON and render
  reproducible PNG charts from that JSON.
- [x] Add per-shape QMM microbenchmarks for Granite's common matrix dimensions.
- [x] Capture and export the stock affine-Q8 Metal kernels used by full
  inference.
- [x] Benchmark uniform G64 against mixed G128 Q8 in interleaved full-input
  runs.
- [x] Prototype and reject symmetric signed Q8 after it measured 2x--13.5x
  slower than stock affine QMM.
- [x] Prototype and reject a specialized BM64 MLX QMM tile after correctness
  failure and no speed improvement; restore the stock kernel.
- [x] Implement exact vocabulary-tiled CTC projection plus running argmax as a
  public-API fallback. It is 1.4% slower and does not lower the full-model peak;
  true QMM-epilogue fusion requires a new MLX primitive.
- [x] Measure FP8 and INT8 activation storage. Native low-precision activation
  matmul is unavailable on the M1 Max stack; emulation is slower and does not
  reduce full-model peak memory.
- [x] Add MLX allocator memory reporting and a configurable recycled-buffer
  cache limit.
- [x] Add exact online-softmax/tiled middle CTC conditioning. Tile 2048 keeps
  the full transcript exact and lowers one-pass peak from 12.64 GB to 9.66 GB
  after disabling cache retention.
- [x] Add block-aligned temporal inference with left/right context, central
  emission cropping, and one global CTC collapse/tokenizer decode.
- [x] Make the bounded 122.88s/10.24s temporal profile and 64 MiB MLX cache the
  CLI defaults, with `--no-chunking` and `--audio-chunk-duration 0` one-pass
  overrides.
- [x] Compare the prior 20.48s context, 10.24s context, and a larger central
  chunk in three rotated long-form rounds. Select 10.24s context after it was
  13.2% faster, used 25 MB less peak memory, and retained 99.9045% word
  agreement with the prior bounded transcript.
- [x] Compare the selected profile directly with matching Q8 FP16 one-pass
  output: 30 edits across 13,615 reference words (99.7797% agreement). Manually
  inspect every changed passage and confirm there are no missing/duplicated
  sentences or catastrophic sections; record the localized technical-word
  differences in the benchmark guide.
- [x] Prototype per-Conformer-layer MLX compilation and retaining the allocator
  cache between chunks. Reject both because neither delivered a useful speed
  improvement without increasing peak memory.
- [x] Benchmark Q8/Q6/Q5 bounded-memory profiles and preserve machine-readable
  results plus PNG charts.
- [ ] Replace context-window approximation with exact layer-wise temporal
  streaming if byte-identical one-pass output is required on mobile.
- [ ] Stream file decoding/resampling into overlapping model windows instead of
  retaining the complete Float32 waveform. The isolated-window measurement
  projects a further 422 MB saving for the 102-minute fixture.
- [ ] Validate bounded Q8 and Q6 profiles on physical iPhone and iPad hardware,
  including jetsam headroom, thermals, battery use, and sustained speed.
- [x] Test short and long audio, generated silence, stereo downmixing, non-16
  kHz resampling, multiple inputs, common audio/video formats, and corrupt
  inputs through unit, opt-in integration, and benchmark fixtures.
- [x] Test published default-model download from an isolated empty cache and local model loading.
- [x] Test cache reuse, isolated partial/corrupt states, interruption/resume,
  coded network/download errors, and first-download progress reporting.

### Native Core ML backend experiment

- [x] Implement a standalone PyTorch Granite graph that loads IBM's checkpoint
  directly and avoids dependency on a matching Transformers model class.
- [x] Convert fixed-shape FP16 Core ML ML Programs with greedy CTC frame IDs as
  output and verify exact short-fixture frame parity.
- [x] Implement `GraniteCoreMLRecognizer` with the shared Swift frontend,
  tokenizer, CTC timestamps, chunk/context handling, progress, cancellation,
  coded errors, timing breakdowns, and persistent compiled-model caching.
- [x] Add `--backend coreml`, `--coreml-model`, and
  `--coreml-compute-units`; make omitted Core ML chunk settings fill the fixed
  graph automatically.
- [x] Measure CPU, CPU+GPU, CPU+ANE, and automatic placement. Select CPU+GPU on
  M1 Max after identifying 32 ANE-unsupported dynamic attention matmuls.
- [x] Test Core ML linear INT8 and uniform/k-means palette weight compression
  across per-tensor and grouped-channel configurations.
- [x] Select macOS 15 uniform Q8 grouped-channel size 1 as the accuracy-first
  quantization profile.
- [x] Search 8,192, 16,384, 24,576, and 32,768 feature-frame graphs. Select
  16,384 frames as the measured speed/memory sweet spot.
- [x] Beat the release MLX backend on the 101m59s fixture: 17.32s versus 24.41s
  speech inference, with 0.265% word disagreement versus MLX output.
- [x] Re-run the complete formatting-enabled CLI path: 19.83s processing,
  24.40s process wall, and 99.65% lexical agreement with formatted MLX output.
- [x] Test and reject the current INT8 activation graph (runtime crash), note
  that Core ML Tools 9 offers no usable FP8 activation conversion for this
  graph, and measure that splitting the final CTC head cannot materially help.
- [x] Check in conversion/quantization/benchmark scripts, JSON results, a PNG
  comparison chart, documentation, unit coverage, and an opt-in real-model CLI
  parity test.
- [x] Publish the selected Core ML package with matching tokenizer assets and
  add it to the CLI's automatic model download/cache-management catalog.
- [ ] Validate the Core ML backend on newer Apple Silicon generations and on
  physical iPhone/iPad hardware before making it the general default.

### Local long-form benchmark corpus

The primary long-form performance fixtures are stored outside this repository:

The source videos are supplied to release tests through
`GRANITE_TEST_LONG_AUDIO` and remain outside Git.

Current fixtures:

- Lecture 1: approximately 101m 59s, WebM/Opus, 48 kHz stereo.
- Lecture 2: approximately 107m 19s, MP4/AAC, 44.1 kHz stereo.

Both recordings are predominantly single-speaker lectures and have local SRT/VTT caption files generated by Parakeet MLX. These captions are not independent ground truth and must not be used for quantitative WER claims. They may still be useful for rough navigation, segment boundaries, and qualitative inspection.

- [ ] Add a dedicated convenience script that extracts mono 16 kHz WAV audio
  through `ffmpeg` without altering the original videos; the reproducible
  extraction command and fixture SHA-256 are already documented.
- [x] Benchmark short clips first, then full lectures.
- [x] Confirm that the Python reference can process the full 101m 59s Lecture 1 input in one pass on MPS.
- [-] Compare one-pass and chunked modes across Apple Silicon memory sizes. M1
  Max one-pass/bounded comparisons are complete; other Macs and physical mobile
  devices remain post-release validation.
- [x] Record model-load time, audio duration, inference time, real-time factor,
  peak memory, output size, stability, and transcript hashes.
- [x] Compare Python reference and Swift output text on identical extracted
  audio, explicitly separating native BF16, runtime FP16, and promoted FP32.
- [x] Add pseudo-reference similarity diagnostics against Parakeet MLX output:
  - [x] Run Parakeet and Granite on identical fixed audio.
  - [x] Normalize casing, punctuation, whitespace, and caption artifacts.
  - [x] Report chrF, BLEU, character similarity, and word overlap where useful.
  - [x] Inspect aggregate results and targeted problematic windows.
  - [x] Use these only to detect catastrophic divergence or conversion/runtime
    mistakes, never to claim WER or human-level correctness.
- [x] Use local Parakeet captions only for navigation and qualitative
  side-by-side inspection, never as quantitative WER ground truth.
- [x] Inspect a short Granite/Parakeet clip using waveform and spectrogram overlays; confirm word onsets are broadly plausible and document the leading-silence hallucination caveat.
- [ ] Obtain independently human-verified transcripts or use a separately curated evaluation dataset before reporting WER.
- [x] Keep generated media outside Git and retain only intentional,
  reproducible benchmark summaries and deterministic transcript artifacts.

## Phase 8: Model publication

- [x] Publish FP16, Q8, Q6, Q5, and Q4 to separate Hugging Face repositories for both the Apache 2.0 and non-commercial source-weight families.
- [x] Preserve each source-weight license and usage restrictions in its repository metadata and model card.
- [x] Include source attribution, conversion command, checksums, precision/quantization, benchmark table, and tested runtime versions in every model card.
- [x] Include only the model artifacts needed by the Swift runtime plus the model card and conversion manifest.
- [x] Record the exact source-model revision and converter SHA-256.
- [x] Make the published Apache Q8 repository the CLI/library default and verify automatic first-use download plus transcription.
- [x] Publish the punctuation formatter FP16/Q8/Q6/Q5/Q4 checkpoints to five
  separate repositories with source attribution and per-repository bolded
  comparison rows, and group them in a dedicated Hugging Face collection.

## Phase 9: CLI release completion

The native ASR and punctuation runtimes, timestamps, batch input loop,
Parakeet-compatible exporters, download/cache UX, stable errors, and
application integration APIs are working. The remaining release work is the
opt-in integration gates and distribution packaging.

### Parakeet-compatible outputs

- [x] Port CTC word timestamps and subtitle grouping from the Python reference.
- [x] Implement native Swift exporters for `txt`, `srt`, `vtt`, `json`, and `all`.
- [x] Implement `--highlight-words` for word-highlighted SRT and WebVTT.
- [x] Wire `--output-dir` into file generation.
- [x] Implement `--output-template`, including `filename`, `parent`, `index`, and `date` variables.
- [x] Include transcript, raw/formatted text, word and segment timestamps, model, activation precision,
  audio duration, load time, transcription time, real-time factor, and other
  performance metadata in JSON.
- [x] Keep progress and diagnostics on stderr so TXT/JSON stdout remains parseable.
- [x] Add deterministic overwrite/collision behavior for generated files.

### Input handling

- [x] Process every supplied input path rather than only `inputs.first`.
- [x] Load the ASR and punctuation models once and reuse them across all inputs in one invocation.
- [x] Handle duplicate basenames and output-path collisions predictably.
- [x] Test WAV, MP3, M4A, FLAC, WebM, and MP4 inputs.
- [x] Test mono/stereo conversion, non-16 kHz resampling, silence, empty audio,
  missing inputs, unsupported formats, and corrupt files.
- [x] If ffmpeg is absent, retain AVFoundation's directly readable formats and
  return coded installation/recovery guidance for unsupported containers.
- [ ] Declare `ffmpeg` as a Homebrew dependency so uncommon audio/video
  containers work consistently when AVFoundation cannot decode them.

### Model download and cache UX

Implementation outline: add a loader-level progress callback carrying model
role, repository, cache path, completion fraction, and throughput; pass the
existing `HubApi.snapshot` `Progress`/bytes-per-second updates through it. The
CLI will render a throttled single-line progress meter on an interactive stderr
terminal and coarse milestone lines on redirected stderr. Before downloading,
it will print the ASR or punctuation repository, resolved cache directory, and
known expected size for the published defaults (falling back to Hub metadata
for other repositories), then preflight available disk space. Warm-cache paths
will report a concise cache hit and skip progress rendering. Download failures
will be classified into authentication, network, disk, integrity, and
incomplete-cache errors. Tests will use an injectable Hub/download client and a
temporary cache to cover completion, cache reuse, interruption/resume, and
corruption without depending on the public service.

- [x] Show first-download progress; the isolated Q8 test took 921 seconds with
  no visible progress, and the isolated 55.8 MB punctuation Q8 download took
  176 seconds with no visible progress.
- [x] Display the model repository, expected download size, and cache destination.
- [x] Test a warm-cache run and verify that no model files are downloaded again.
- [x] Test interrupted downloads and resumability with an isolated opt-in
  network integration gate; partial files are retained and a resumed download
  reaches a complete cache.
- [-] Add clear network, insufficient-disk-space, checksum, incomplete-download,
  and corrupted-cache errors: coded disk/download/incomplete/corrupt diagnostics
  and repair instructions are implemented; repository checksum verification
  remains where authoritative checksums are available.
- [x] Expose optional Hugging Face authentication for private/gated model IDs.
- [x] Keep all download progress and diagnostics on stderr.
- [x] Add `granite-mlx models list|download|remove`, including the complete
  published catalog, short aliases, actual installed sizes, JSON listing,
  cache-safe model detection, confirmation, `--yes`, and `remove --all`.
- [x] Display `[ ]` for absent, `[x]` for complete, and `[-]` for partial cache
  entries, with repair and removal commands for partial entries.
- [x] Add a platform-independent `GraniteModelStorage` API covering materialized
  repositories, the shared Hugging Face transfer cache, and compiled Core ML
  artifacts. Thread it through model browsing, download, removal, MLX loading,
  formatting, and Core ML loading; retain environment variables only as optional
  CLI defaults.
- [x] Add a storage-bound `GraniteModelManager` suitable for macOS, iOS, and
  iPadOS model-management interfaces, with catalog browsing, state inspection,
  progress/cancellation-aware download, installed-model listing, and removal.

### CLI polish and stable behavior

- [x] Add `granite-mlx --version` (currently reports the provisional `0.1.0`; release automation must become its source of truth).
- [x] Validate `--output-format`, numeric subtitle/chunk settings, and `all` without an output directory.
- [x] Add common transcription, video, export, raw-output, and model-management
  examples directly to root and transcribe help.
- [x] Retain Parakeet's default SRT output when no output directory is supplied;
  multiple stdout results use JSON Lines so file boundaries remain unambiguous.
- [x] Give validation and runtime failures stable `GMLX-*` diagnostic codes and
  technical context; multi-input batches continue safely and exit nonzero if
  any input failed.
- [x] Map Ctrl-C to cooperative library cancellation, terminate ffmpeg, write
  outputs atomically, and remove files from an interrupted multi-format export.

### Release-level tests

- [x] Add CLI integration tests for TXT, SRT, VTT, JSON, and `all`.
- [x] Add unit fixtures for deterministic CTC token/word timestamps, formatter
  sentence-boundary segmentation, SRT/VTT formatting, and highlighted words.
- [x] Add multiple-input, duplicate-name, output-directory, output-template,
  and collision tests.
- [x] Add isolated model first-download, warm-cache, interrupted/resumed-download,
  partial-cache, and corrupt-cache tests.
- [x] Add generated WAV stereo/downmix/resampling tests and MP3, M4A, FLAC,
  WebM, and MP4 ffmpeg-backed format tests.
- [x] Add an opt-in long-form bounded-memory regression test with exact
  checkpoint-matched transcript comparison and an MLX peak-memory threshold.
- [x] Test the published archive in an isolated clean-install environment with
  no Python or ffmpeg on `PATH`; retain a separate physical clean-Mac test for
  the Homebrew formula.
- [x] Verify machine-readable stdout never contains progress or diagnostic text.

## Phase 10: GitHub repository and releases

### Repository preparation

- [x] Add Apache-2.0 and MIT software licenses as a dual-license choice,
  explicitly separate from model/checkpoint licenses.
- [x] Pin known-good `mlx-swift`, `swift-transformers`, and
  `swift-argument-parser` versions/revisions.
- [x] Document the minimum macOS, Apple Silicon, Swift, and Xcode requirements.
- [x] Document the `ffmpeg` dependency, model cache location, default Q8 model,
  alternate model selection, output formats, and first-run download behavior.
- [x] Review benchmark artifacts and generated transcripts; keep only intended,
  reproducible project files in Git.
- [x] Review the repository for local absolute paths, caches, downloaded weights,
  credentials, and temporary files.
- [x] Add and validate a DocC catalog, document every exported Swift declaration,
  and add a repeatable warnings-as-errors documentation check.
- [x] Create local implementation and release-hardening commits.
- [x] Create the `kylehowells/Granite-MLX` GitHub remote and push the repository.

### CI and GitHub Releases

- [x] Add GitHub Actions for `swift test`, `swift build -c release`, macOS
  Apple Silicon validation, and an iOS Simulator library build.
- [x] Build/package the required MLX Metal library as part of the release workflow.
- [x] Add a repeatable release script that produces a versioned archive containing
  the executable, required runtime assets, license, and concise installation notes.
- [x] Create and push semantic-version tags `0.1.0` and the documentation-fix
  release `0.1.1`.
- [x] Publish the ARM64 archive and SHA-256 checksum in the `0.1.1` GitHub Release.
- [x] Independently download, checksum, inspect, and exercise the published
  `0.1.1` archive in an isolated clean-install environment.
- [x] Confirm the release runs without Python and downloads the default model
  separately rather than bundling model weights.

## Phase 11: Homebrew distribution

- [ ] Create or update the `kylehowells/homebrew-tap` repository.
- [ ] Add `Formula/granite-mlx.rb` referencing the tagged GitHub release and checksum.
- [x] Select a binary formula using the tagged macOS ARM64 release archive.
- [x] Package the compatible MLX Metal library inside the release archive.
- [ ] Declare Apple Silicon, minimum macOS, Swift/Xcode build requirements, and
  the `ffmpeg` runtime dependency.
- [ ] Add formula tests for `granite-mlx --version`, `granite-mlx --help`, and
  offline `granite-mlx models list`.
- [-] Do not put real transcription in `brew test`: model weights are deliberately
  downloaded separately, and formula tests must remain model-free.
- [-] A `--build-from-source` formula path is not part of the binary `0.1.0`
  distribution; source builds remain available directly through SwiftPM.
- [ ] Test normal tap installation, upgrade, uninstall, and reinstall behavior.
- [ ] Verify end-to-end installation and first transcription:

```bash
brew install kylehowells/tap/granite-mlx
granite-mlx sample.m4a
```

Do not bundle model weights into the Homebrew formula. Homebrew installs the
CLI; the CLI downloads and caches the default 466 MiB Q8 model separately.

## Release gates

Before calling the project complete:

- [x] PyTorch and MLX final outputs agree within documented transcript
  tolerances; intermediate tensor fixtures remain optional follow-up work.
- [x] Swift and Python reference implementations produce matching transcripts on the fixed fixture.
- [x] Short and long audio paths work reliably on the release-development M1
  Max, including exact long-form bounded regression coverage.
- [x] The CLI has deterministic TXT and JSON output for fixed model/audio
  fixtures.
- [x] The published archive installs and runs in an isolated clean-install
  environment without a Python runtime; the Homebrew formula will additionally
  be tested on a separate clean Mac.
- [x] The Granite-MLX software license is documented separately from any model/checkpoint license.
- [x] GitHub release installation works.

## Current implementation checkpoint (2026-08-28)

Completed: native 16-layer Granite 5.0 Conformer/CTC inference, exact
torchaudio-compatible frontend, original and converted checkpoint loading,
greedy CTC/BPE decoding, AVFoundation/ffmpeg audio and video input, FP16 and
FP32 converted artifacts, transcript parity on the fixed 20-second fixture,
affine Q8/Q6/Q5/Q4/Q3/Q2 artifacts, strict packed-tensor validation, and
machine-readable benchmark output. The 101m 59s lecture was benchmarked in one
pass for every precision. Q8 is near-lossless relative to Swift FP32, Q4 has
1.51% transcript disagreement, Q3 degrades materially, and Q2 fails to decode
words. Full results live in `Benchmarks/quantization/results.json`.

The Q8-native optimization pass is also complete. The recommended runtime is a
mixed G128/G64 affine-Q8 checkpoint with FP16 activations. It transcribes the
101m 59s lecture in a paired median 20.25 seconds (302x real time) on the M1
Max, approximately 2% faster than uniform G64. Full machine-readable results
and PNG charts live in `Benchmarks/q8-optimization`.

The model runtime, publication work, and intended `0.1.0` CLI feature surface
are complete. Ten converted MLX checkpoints are published under `iky1e`: FP16,
Q8, Q6, Q5, and Q4 for both source-weight families. The selected Apache Core ML
Q8 package is published separately. Full-lecture agreement results and
publication metadata live in `Benchmarks/model-publication` and
`Benchmarks/coreml`; Apache MLX Q8 remains the general default. The
release-critical path is now the remaining opt-in integration gates, GitHub
Releases, and Homebrew packaging.

The Swift CLI now processes every input while reusing one loaded ASR model and
one loaded formatter. It derives CTC token/word times at 12.5 fps, maps native
formatter sentence ranges onto timed words, applies subtitle safety limits,
and exports TXT, SRT, Parakeet-style WebVTT, JSON, or all four. Output
directories, templates, highlighted words, duplicate-path collision suffixes,
raw/formatted JSON fields, `--version`, and argument validation are wired up.
Unit tests cover timing, segmentation, subtitle rendering, cache management,
audio conversion, model validation, diagnostics, and cancellation. The
integration suite covers all exporters, collisions, stdout/stderr separation,
download interruption, Ctrl-C, and bounded-memory long-form transcription,
with network and model-heavy gates explicitly opt-in. A real 20-second Q8/Q8
smoke test generated all four files successfully.

The public README, command help, benchmark guides, implementation plan, test
guide, and DocC catalog have been audited for machine-specific paths and broken
relative links. Every source-declared public API carries `///` documentation.
DocC reports 100% type/global coverage and 93% member coverage because it also
counts 21 compiler-synthesized Codable and RawRepresentable initializers that
have no source declaration to annotate.

The download UX is now implemented. Both automatic first-use downloads and
explicit `models download` operations report repository, role, expected size,
cache path, completion, and throughput on stderr; warm cache hits skip network
access. `models list` reports the complete 16-model catalog, defaults, expected
or actual sizes, cache status, and JSON; `models remove` supports confirmation,
`--yes`, and Granite-only `--all`. Architecture-aware cache scanning prevents
the manager from listing or deleting unrelated Swift Hub models. A fresh
punctuation-Q4 download was observed through completion and then removed again,
and cache-hit, noninteractive-removal protection, debug tests, release build,
and cached default transcription were verified.

The optional native Core ML backend is also working. A macOS 15 Q8
palettized, per-channel 16,384-frame graph runs the full 101m59s lecture in
17.32 seconds of speech inference on the M1 Max, 29% faster than the 24.41s
then-current bounded MLX release profile. It uses 2.10 GB peak footprint rather
than 1.64 GB and differs from the MLX transcript by 0.265% of reference words. The
formatting-enabled path completes processing in 19.83 seconds and retains
99.65% lexical agreement with formatted MLX output. Reproducible scripts,
negative ANE/INT8 results, machine-readable measurements, and a PNG comparison
are in `Benchmarks/coreml`. The selected package is published as
`iky1e/granite-speech-5.0-470m-turboctc-coreml-q8`, integrated with automatic
download, list, compiled-cache accounting, and removal, and selectable as the
persistent user default through `granite-mlx config set backend coreml`. The
setting is stored in the user's Application Support configuration rather than
an environment variable; explicit `--backend` still wins. MLX remains the
general default pending validation on newer Macs and physical iPhone/iPad
hardware.

The MLX default has since moved from 20.48 to 10.24 seconds of context. In a
separate three-round paired experiment under heavy background load, that cut
median MLX Q8 inference from 25.281 to 21.952 seconds and peak footprint from
1.652 to 1.627 GB. Those absolute timings are not directly comparable to the
quieter Core ML release run; a future quiet cross-backend release matrix should
measure the new default directly. Against matching Q8 FP16 one-pass output, the
selected bounded profile has 30 edits across 13,615 words (99.7797% agreement).
Manual review found only localized word/filler changes, no missing or duplicated
passages, and one newly introduced technical-word regression (`dot` → `start`).
- [ ] Homebrew formula installation works.
- [x] Default model download and cache behavior are documented.

## Recommended implementation order

1. Repository/package skeleton.
2. Python reference and conversion tooling.
3. MLX Python numerical parity.
4. Swift frontend and model runtime.
5. Basic offline CLI.
6. Output formats and long-audio handling.
7. Tests and benchmarks.
8. Hugging Face model publication.
9. Complete the native Swift CLI output/input/download surface.
10. Prepare GitHub, CI, and the first tagged release.
11. Publish and validate the Homebrew formula.
