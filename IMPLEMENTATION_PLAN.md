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
- [ ] Treat `mlx-swift-lm` as a reference for package/model-loading conventions, but implement Granite as a custom encoder rather than an LLM registry entry.
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
- [ ] Save deterministic intermediate artifacts for a short fixture:
  - [ ] frontend features
  - [ ] encoder outputs
  - [ ] middle self-conditioned output
  - [ ] final logits
  - [ ] greedy CTC token IDs
  - [ ] decoded transcription
- [ ] Record the exact Transformers revision and model revision used for validation.

## Phase 3: MLX conversion

- [x] Add `Scripts/convert_granite.py`.
- [x] Accept a source model ID/path as an argument so either Granite 5.0 weight-license variant can be converted.
- [x] Convert the original safetensors checkpoint into MLX-loadable safetensors.
- [x] Preserve or generate the required `config.json`, tokenizer files, processor metadata, and model manifest.
- [x] Remove training-only state such as BatchNorm `num_batches_tracked` values.
- [ ] Verify every tensor name and shape against the Swift module tree.
- [x] Document all layout changes, especially linear and convolution weight transpositions.
- [ ] Produce at least:
  - [x] BF16 or FP16 baseline checkpoint
  - [x] FP32 validation checkpoint if needed
  - [x] affine Q8/Q6/Q5/Q4/Q3/Q2 checkpoints after numerical parity was established
- [x] Store quantization mode, bit width, and group size in `config.json`.
- [x] Validate packed weight, scale, and quantization-bias shapes at load time.
- [ ] Add a conversion README explaining source model, output format, command, and expected sizes.

## Phase 4: MLX numerical parity

- [ ] Implement an MLX Python model matching the Granite 5.0 architecture.
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
- [ ] Add `Scripts/compare_pytorch_mlx.py`.
- [ ] Compare PyTorch and MLX intermediate tensors with tolerances appropriate to BF16/FP16.
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
- [ ] Add timestamp-aware chunk overlap merging rather than text-only merging.
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
- [ ] `--chunk-duration` and `--overlap-duration` for long audio.
- [x] `--highlight-words` for word-highlighted SRT/VTT.
- [x] `--max-words`, `--silence-gap`, and `--max-duration` for subtitle segmentation.
- [ ] `--bf16` / `--fp32`.
- [x] `--verbose`.
- [x] `--version` (provisional `0.1.0`; derive it from release metadata before tagging).
- [x] `--help` generated by ArgumentParser (more examples remain for final CLI polish).

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
- [ ] Add model-loading tests using a small or test-only checkpoint where practical.
- [ ] Add PyTorch-vs-MLX regression tests using fixed audio fixtures.
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
- [x] Make the bounded 122.88s/20.48s temporal profile and 64 MiB MLX cache the
  CLI defaults, with `--no-chunking` and `--audio-chunk-duration 0` one-pass
  overrides.
- [x] Benchmark Q8/Q6/Q5 bounded-memory profiles and preserve machine-readable
  results plus PNG charts.
- [ ] Replace context-window approximation with exact layer-wise temporal
  streaming if byte-identical one-pass output is required on mobile.
- [ ] Stream file decoding/resampling into overlapping model windows instead of
  retaining the complete Float32 waveform. The isolated-window measurement
  projects a further 422 MB saving for the 102-minute fixture.
- [ ] Validate bounded Q8 and Q6 profiles on physical iPhone and iPad hardware,
  including jetsam headroom, thermals, battery use, and sustained speed.
- [ ] Test short audio, long audio, silence, stereo input, non-16 kHz input, and multiple files (20 s and 10 min audio/video paths are complete).
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

- [ ] Add a benchmark script that extracts mono 16 kHz WAV audio through `ffmpeg` without altering the original videos.
- [ ] Benchmark short clips first, then full lectures.
- [x] Confirm that the Python reference can process the full 101m 59s Lecture 1 input in one pass on MPS.
- [ ] Compare one-pass and chunked modes across available Apple Silicon memory sizes.
- [ ] Record model-load time, audio duration, inference time, real-time factor, peak memory, and output size.
- [ ] Compare Python reference and Swift output text on identical extracted audio.
- [ ] Add a pseudo-reference similarity diagnostic against Parakeet MLX transcripts:
  - [ ] Run Parakeet and Granite on identical fixed audio windows.
  - [ ] Normalize casing, punctuation, whitespace, and obvious caption artifacts.
  - [ ] Report chrF and BLEU (BLEU is intended here; not "Blue") plus simple character/word overlap.
  - [ ] Compare per-window distributions as well as the aggregate score.
  - [ ] Use this only to detect catastrophic divergence or conversion/runtime mistakes, not to claim WER or model accuracy.
- [ ] Use the local captions only for rough navigation and qualitative side-by-side inspection, never as quantitative WER ground truth.
- [x] Inspect a short Granite/Parakeet clip using waveform and spectrogram overlays; confirm word onsets are broadly plausible and document the leading-silence hallucination caveat.
- [ ] Obtain independently human-verified transcripts or use a separately curated evaluation dataset before reporting WER.
- [ ] Keep generated WAVs, transcripts, and benchmark results outside Git or under ignored output directories.

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
- [-] Test mono/stereo conversion, non-16 kHz resampling, silence, empty audio,
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
- [-] Test interrupted downloads and resumability: cancellation is wired through
  the Hub task and partial files are retained for repair/resume; a deterministic
  interrupted-transfer integration fixture remains.
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
- [ ] Test a clean Mac installation and transcription without Python installed.
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
- [ ] Create the `kylehowells/Granite-MLX` GitHub remote and push the repository.

### CI and GitHub Releases

- [ ] Add GitHub Actions for `swift test`, `swift build -c release`, and macOS
  Apple Silicon validation.
- [ ] Build/package the required MLX Metal library as part of the release workflow.
- [ ] Add a repeatable release script that produces a versioned archive containing
  the executable, required runtime assets, license, and concise installation notes.
- [ ] Create the first semantic-version tag, provisionally `0.1.0`.
- [ ] Publish the archive and SHA-256 checksum in a GitHub Release.
- [ ] Download and test the published release archive on a clean Mac.
- [ ] Confirm the release runs without Python and downloads the default model
  separately rather than bundling model weights.

## Phase 11: Homebrew distribution

- [ ] Create or update the `kylehowells/homebrew-tap` repository.
- [ ] Add `Formula/granite-mlx.rb` referencing the tagged GitHub release and checksum.
- [ ] Decide whether the formula builds from the tagged SwiftPM source archive or
  installs a release binary; validate the selected approach on a clean machine.
- [ ] Build the MLX Metal library during formula installation or package the
  compatible compiled runtime asset with the release.
- [ ] Declare Apple Silicon, minimum macOS, Swift/Xcode build requirements, and
  the `ffmpeg` runtime dependency.
- [ ] Add formula tests for `granite-mlx --version` and `granite-mlx --help`.
- [ ] Add a small offline transcription formula test where practical, without
  requiring the default model download during `brew test`.
- [ ] Test `brew install --build-from-source`.
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

- [ ] PyTorch and MLX outputs agree within documented tolerances.
- [x] Swift and Python reference implementations produce matching transcripts on the fixed fixture.
- [ ] Short and long audio paths work reliably.
- [ ] The CLI has deterministic TXT and JSON output.
- [ ] A clean Mac can install and run the CLI without a Python runtime.
- [x] The Granite-MLX software license is documented separately from any model/checkpoint license.
- [ ] GitHub release installation works.

## Current implementation checkpoint (2026-08-27)

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
bounded MLX release profile. It uses 2.10 GB peak footprint rather than 1.64
GB and differs from the MLX transcript by 0.265% of reference words. The
formatting-enabled path completes processing in 19.83 seconds and retains
99.65% lexical agreement with formatted MLX output. Reproducible scripts,
negative ANE/INT8 results, machine-readable measurements, and a PNG comparison
are in `Benchmarks/coreml`. The selected package is published as
`iky1e/granite-speech-5.0-470m-turboctc-coreml-q8`, integrated with automatic
download, list, compiled-cache accounting, and removal, and selectable as the
shell default through `GRANITE_MLX_BACKEND=coreml`. MLX remains the general
default pending validation on newer Macs and physical iPhone/iPad hardware.
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
