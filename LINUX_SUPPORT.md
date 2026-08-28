# Linux and NVIDIA CUDA Support Plan

## Status

Linux support is a goal for a release after the initial macOS `0.1.0` release.
It is not part of the first Homebrew release and must not delay that release.

The intended end state is a native Swift `granite-mlx` CLI for 64-bit Linux
with NVIDIA CUDA acceleration. A CPU-only Linux build is a useful portability
and CI milestone, but it is not expected to provide performance comparable to
Apple Silicon or CUDA.

This document records the anticipated work and release gates. MLX's Linux and
CUDA support is evolving quickly, so dependency capabilities and requirements
must be rechecked against current upstream documentation when implementation
begins.

## Current upstream position

MLX supports CPU execution on Linux and has a CUDA backend for NVIDIA GPUs.
Current installation and source-build requirements are documented by upstream:

- [MLX build and installation guide](https://ml-explore.github.io/mlx/build/html/install.html)
- [MLX repository](https://github.com/ml-explore/mlx)
- [MLX Swift repository](https://github.com/ml-explore/mlx-swift)

At the time this plan was written:

- MLX offers Linux CPU and CUDA 12/13 backends.
- The CUDA backend requires a compatible NVIDIA GPU, driver, CUDA toolkit,
  Linux distribution, and glibc version.
- MLX Swift can build native Linux examples using CMake.
- MLX Swift's CUDA-enabled route uses CMake with `MLX_BUILD_CUDA=ON`.
- The MLX Swift 0.31.4 SwiftPM Linux configuration used by Granite-MLX selects
  the CPU backend and excludes the CUDA sources.

Consequently, a CPU-only SwiftPM build and a GPU-accelerated CUDA build are
different milestones. Enabling Linux in `Package.swift` alone does not produce
a CUDA-enabled CLI.

## Scope

The first Linux release should provide:

- A native Swift command-line executable for x86-64 Linux.
- NVIDIA CUDA acceleration through MLX.
- The existing MLX Granite Speech architecture and affine checkpoints.
- Automatic Hugging Face model download and cache management.
- WAV input without external media dependencies.
- General audio and video input through `ffmpeg`.
- TXT, SRT, WebVTT, JSON, and combined output modes.
- Punctuation and capitalization through the existing MLX formatter.
- Progress, cancellation, structured errors, and model-management commands.
- Output parity within documented tolerances against the macOS/Python
  references.

The Linux release will not provide:

- Core ML.
- AVFoundation media decoding.
- Apple Metal execution or `mlx.metallib`.
- Apple-specific configuration or cache paths.
- A promise that every NVIDIA GPU generation has identical performance or
  supports every experimental quantization format.

## Phase 1: Revalidate upstream support

- [ ] Select and pin a current MLX and MLX Swift release with tested Linux CUDA
  support.
- [ ] Record supported Linux distributions, architectures, glibc versions,
  CUDA toolkit versions, driver versions, and NVIDIA compute capabilities.
- [ ] Audit the CUDA backend for every operation used by Granite speech and the
  punctuation formatter.
- [ ] Confirm whether missing operations fail, fall back to CPU, or can be
  assigned explicitly to a CPU stream.
- [ ] Confirm Linux support in Swift Transformers, Hub, Tokenizers, Swift
  Argument Parser, and every transitive dependency.
- [ ] Decide whether upstream MLX Swift has gained a supported CUDA-enabled
  SwiftPM path by the time work starts.
- [ ] Open or track upstream issues for any missing Swift/CUDA functionality
  rather than maintaining unnecessary permanent forks.

Operations requiring explicit validation include:

- affine quantized matrix multiplication;
- ordinary matrix multiplication and batched matrix multiplication;
- convolution and depthwise convolution;
- block attention and relative-position operations;
- softmax and reductions;
- padding, slicing, concatenation, reshaping, and transposition;
- embedding lookup;
- safetensors loading;
- FP16 and FP32 evaluation;
- explicit synchronization, stream selection, and memory reporting.

## Phase 2: Make Granite-MLX platform-neutral

### Package and source layout

- [ ] Separate the portable Granite runtime from Apple-only backends.
- [ ] Add conditional compilation for Apple and Linux platform facilities.
- [ ] Ensure the portable `GraniteMLX` product does not unconditionally import
  AVFoundation, Core ML, Metal, AppKit, or Darwin.
- [ ] Decide whether Core ML should remain in the main library behind
  conditional compilation or move to a separate Apple-only target/product.
- [ ] Keep one high-level transcription API where practical while exposing
  platform-specific backend availability clearly.
- [ ] Make unsupported backends fail at argument validation rather than later
  during model loading.

### Audio and video input

- [ ] Preserve the native WAV parser as the dependency-free Linux path.
- [ ] Use `ffmpeg`/`ffprobe` for other Linux audio and video containers.
- [ ] Do not attempt to compile AVFoundation code on Linux.
- [ ] Preserve mono downmixing, normalization, resampling to 16 kHz, progress,
  cancellation, process termination, and coded diagnostics.
- [ ] Replace macOS-specific `brew install ffmpeg` recovery text with
  distribution-appropriate Linux guidance.
- [ ] Test paths containing spaces, Unicode, shell metacharacters, and broken
  symlinks without invoking a shell to launch `ffmpeg`.

### Operating-system services

- [ ] Replace unconditional `import Darwin` with `Darwin`/`Glibc` platform
  imports.
- [ ] Verify SIGINT handling and cooperative cancellation on Linux.
- [ ] Define XDG-compliant configuration and cache locations, respecting
  `XDG_CONFIG_HOME` and `XDG_CACHE_HOME`.
- [ ] Preserve explicit test/cache overrides such as
  `GRANITE_MLX_HUB_DIRECTORY`.
- [ ] Verify filesystem capacity checks, atomic file replacement, permissions,
  symlink safety, and interrupted-download handling on Linux filesystems.
- [ ] Review terminal detection and progress rendering on common Linux shells,
  terminals, pipes, containers, and systemd logs.
- [ ] Audit Foundation APIs for Linux behavior differences.

### CLI behavior

- [ ] Hide Core ML models, configuration, and flags when unavailable, or report
  a precise platform-availability error.
- [ ] Keep MLX as the Linux backend default.
- [ ] Preserve CLI syntax and output compatibility across macOS and Linux.
- [ ] Add platform/backend information to `--version`, verbose diagnostics, and
  benchmark JSON where useful.
- [ ] Ensure examples and errors do not assume Homebrew or macOS paths.

## Phase 3: Establish a CPU-only Linux build

The CPU build is primarily a portability and correctness milestone. It allows
ordinary hosted CI to compile and run model-free tests without requiring a GPU.

- [ ] Build Granite-MLX with SwiftPM and MLX Swift's Linux CPU backend.
- [ ] Add an Ubuntu container or reproducible development environment with
  Swift, BLAS, LAPACK, LAPACKE, OpenBLAS, `gfortran`, and `ffmpeg`.
- [ ] Compile the library and CLI on x86-64 Linux.
- [ ] Run all model-free library and CLI tests.
- [ ] Run a short real-model CPU transcription to prove end-to-end correctness.
- [ ] Compare its raw and formatted transcript with fixed reference output.
- [ ] Measure CPU runtime and memory without presenting it as the recommended
  Linux performance configuration.
- [ ] Determine whether Linux ARM64 CPU builds are useful and supportable or
  should remain an unpromised development configuration.

## Phase 4: Select the Swift CUDA integration strategy

Choose one supported build route after reevaluating upstream MLX Swift:

### Preferred: upstream CUDA-enabled SwiftPM

- [ ] Use an upstream-supported CUDA-enabled SwiftPM configuration if one is
  available and maintained.
- [ ] Avoid forking MLX Swift solely for build-system differences.

### Alternative: CMake-built MLX linked to Swift

- [ ] Build MLX/MLX Swift with CMake using `MLX_BUILD_METAL=OFF` and
  `MLX_BUILD_CUDA=ON`.
- [ ] Export a stable library and module interface that Granite-MLX can link.
- [ ] Teach Granite-MLX's build/release scripts to locate those artifacts.
- [ ] Record all required runtime shared libraries and loader paths.
- [ ] Ensure the final package does not depend on a developer checkout or
  machine-specific absolute paths.

### Last resort: narrowly maintained fork

- [ ] If required, keep any MLX Swift fork minimal and submit generally useful
  changes upstream.
- [ ] Pin the exact fork revision and document the reason for every divergence.
- [ ] Define an exit plan for returning to upstream releases.

The selected strategy must support clean builds in containers and release CI.

## Phase 5: Bring up Granite on CUDA

- [ ] Add explicit device selection and verify that Linux CUDA runs on the GPU,
  not an accidental CPU fallback.
- [ ] Load the existing converted FP16 checkpoint and run the fixed short audio
  fixture.
- [ ] Compare frontend features, frame IDs, raw text, word timings, and formatted
  text with the established references.
- [ ] Bring up the existing affine Q8 checkpoint next.
- [ ] Validate Q6, Q5, and Q4 after Q8 is correct.
- [ ] Confirm mixed G128/G64 quantization metadata and tensor layouts are
  accepted by CUDA kernels.
- [ ] Verify bounded long-form chunking, context cropping, one global CTC
  collapse, and punctuation mapping.
- [ ] Audit every explicit MLX memory/cache call for CUDA semantics.
- [ ] Add CUDA device name, compute capability, driver, toolkit, backend, and
  MLX version to benchmark diagnostics.
- [ ] Produce precise coded errors for absent drivers, unsupported GPUs,
  unavailable operations, out-of-memory failures, and incompatible CUDA
  libraries.

## Quantization considerations

Existing Granite checkpoints use MLX affine weight-only quantization. They
should remain the first Linux/CUDA target because they already have measured
accuracy and known output on macOS.

- [ ] Verify affine Q8 correctness and performance before introducing another
  weight format.
- [ ] Check whether CUDA kernels support every affine bit width and group size
  used by published Granite and punctuation checkpoints.
- [ ] Do not assume a smaller checkpoint is faster; benchmark each format and
  matrix shape on representative NVIDIA GPUs.
- [ ] Keep activation precision separate from weight quantization in the CLI,
  documentation, and benchmark data.

Current MLX APIs also expose `mxfp4`, `mxfp8`, and `nvfp4`. These may use
NVIDIA-oriented block-scaled matrix operations on supported hardware, but the
current Granite speech loader intentionally accepts only affine quantization.
Supporting them would require separate work:

- [ ] Add converter support and complete checkpoint metadata.
- [ ] Extend loader validation for scales, optional biases, global scales,
  group sizes, tensor layouts, and backend requirements.
- [ ] Add model-layer construction for each new quantization mode.
- [ ] Establish FP16/FP32 output baselines before measuring accuracy loss.
- [ ] Benchmark short and full-lecture speed, peak GPU memory, host memory,
  checkpoint size, and transcript disagreement.
- [ ] Test on hardware that actually accelerates the selected format.
- [ ] Publish CUDA-oriented checkpoints separately only if they provide a
  meaningful advantage.
- [ ] Do not replace the portable affine default solely because a format has an
  NVIDIA-oriented name.

## Phase 6: Testing and benchmarks

### Ordinary Linux CI without a GPU

- [ ] Compile the CPU-only library and CLI.
- [ ] Run CTC, tokenizer, subtitle, formatter-safety, output, cache, progress,
  cancellation, error, configuration, and CLI parsing tests.
- [ ] Test native WAV input and generated `ffmpeg` media formats.
- [ ] Verify Core ML is absent or rejected cleanly.
- [ ] Verify model-management commands do not download large checkpoints.
- [ ] Run formatting, lint, and documentation checks that support Linux.

### CUDA CI or manually provisioned GPU runners

- [ ] Run a minimal MLX CUDA operation smoke test before Granite tests.
- [ ] Confirm the selected GPU and prevent silent CPU execution.
- [ ] Run the fixed short Granite FP16 and Q8 transcription fixtures.
- [ ] Compare deterministic token IDs and normalized transcript output.
- [ ] Run punctuation formatter parity.
- [ ] Record driver, toolkit, GPU, compute capability, MLX, Swift, and Linux
  versions with every result.
- [ ] Keep full model downloads out of ordinary pull-request CI; use a cached,
  explicitly triggered GPU workflow or a controlled release runner.

### Release benchmarks

- [ ] Benchmark at least one datacenter GPU and one commonly available consumer
  GPU where practical.
- [ ] Measure cold model load, warm load, speech inference, formatting,
  end-to-end wall time, real-time factor, GPU memory, host RSS, and power where
  accessible.
- [ ] Run the fixed short sample and complete long-form lecture.
- [ ] Compare CUDA output with the macOS Q8 and Python reference outputs.
- [ ] Investigate every large disagreement and retain reproducible extracted
  clips for regressions.
- [ ] Avoid WER claims until independently verified ground truth is available.

## Phase 7: Linux packaging

A CUDA release is unlikely to be one completely self-contained universal
binary because CUDA toolkit, driver, GPU architecture, and shared-library
compatibility must be considered.

- [ ] Decide the initial supported Linux distribution and architecture matrix.
- [ ] Decide whether to distribute tar archives, Debian/RPM packages,
  containers, or a combination.
- [ ] Consider a container image as the first CUDA distribution because it can
  pin user-space CUDA dependencies while relying on the host NVIDIA driver.
- [ ] Do not bundle model weights; retain first-use Hugging Face downloads.
- [ ] Include licenses, version metadata, checksums, and CUDA requirements.
- [ ] Verify runtime library discovery without `LD_LIBRARY_PATH` hacks where
  possible.
- [ ] Test installation into a clean Linux VM/container without Python.
- [ ] Test with read-only application directories and writable XDG cache paths.
- [ ] Test install, upgrade, uninstall, cache retention, and cache removal.
- [ ] Consider Homebrew on Linux only after the native Linux artifact is stable;
  it should not be the only supported Linux installation route.

## Release gates

A Linux CUDA release is ready only when all of the following are true:

- [ ] A clean documented environment builds the same source revision.
- [ ] The CLI runs natively in Swift without Python.
- [ ] CUDA execution is confirmed on supported NVIDIA hardware.
- [ ] Existing affine FP16/Q8 checkpoints load without conversion at runtime.
- [ ] Short-fixture token IDs and transcripts meet documented parity limits.
- [ ] Long-form bounded-memory transcription completes without missing or
  duplicated sections.
- [ ] Formatting remains non-destructive and produces no tokenizer `<unk>`
  leakage.
- [ ] Peak GPU and host memory fit the documented minimum hardware.
- [ ] Every distributed artifact is tested after download in a clean
  environment.
- [ ] Unsupported GPUs, drivers, toolkits, and unavailable backends produce
  actionable coded errors.
- [ ] Installation, model download, cache reuse, interruption, and removal have
  been exercised end to end.
- [ ] Linux limitations and differences from macOS are clearly documented.

## Deferred and optional work

The following should not block the first Linux CUDA release unless testing
shows they are required:

- AMD ROCm or other non-NVIDIA GPU backends.
- Windows support.
- Linux ARM64 CUDA artifacts.
- NVIDIA-specific FP4/FP8 Granite checkpoints.
- Multi-GPU or distributed transcription.
- Streaming audio capture from platform sound APIs.
- Server, daemon, or REST API modes.
- Exact layer-wise streaming instead of the existing bounded context-window
  implementation.

## Suggested release sequence

1. Ship and stabilize macOS `0.1.0`.
2. Land source-level Linux portability without changing macOS behavior.
3. Add CPU-only Linux compilation and model-free CI.
4. Prove one short CPU transcription for correctness.
5. Integrate MLX Swift's CUDA build path.
6. Prove FP16 and affine Q8 CUDA transcription parity.
7. Benchmark representative NVIDIA hardware and complete long-form testing.
8. Publish an experimental Linux CUDA prerelease.
9. Stabilize installation, diagnostics, and the supported hardware matrix.
10. Promote Linux CUDA to a supported Granite-MLX release target.
