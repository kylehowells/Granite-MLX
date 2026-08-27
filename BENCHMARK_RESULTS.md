# Granite-MLX benchmark results

> Historical one-pass baseline. The production CLI now uses bounded temporal
> chunks by default, so the large one-pass memory figures below do not describe
> its normal behavior. See
> [`Benchmarks/memory-optimization`](Benchmarks/memory-optimization) for the
> original bounded-memory experiments,
> [`Benchmarks/bounded-profile-optimization`](Benchmarks/bounded-profile-optimization)
> for the current 122.88/10.24-second profile, and
> [`Benchmarks/backend-matrix`](Benchmarks/backend-matrix) for the current
> three-round Python, Swift MLX, and Core ML comparison.

Measured 26 August 2026 on a 10-core Apple M1 Max MacBook Pro with 64 GB RAM,
macOS 26.5.2, Swift 6.2.3, MLX Swift 0.31.4, Python 3.10.15, PyTorch 2.13.0,
and Transformers 5.15.1. Swift measurements use the release build.

The three Swift rows all use MLX for computation:

- `source BF16` loads IBM's downloaded safetensors checkpoint directly and
  normalizes incompatible tensor layouts in memory.
- `converted FP16` loads a pre-normalized MLX checkpoint with FP16 tensors.
- `converted FP32` loads the same pre-normalized checkpoint with FP32 tensors.

The historical Python row promoted the checkpoint's BF16 values to FP32 at
runtime. It is therefore a **promoted-FP32 reference**, not an original-FP32
checkpoint. Promoting BF16 cannot restore precision that was absent from the
published source weights. Fresh native-BF16, runtime-FP16, and promoted-FP32
Python measurements are preserved in
[`Benchmarks/coreml/python-source-results.json`](Benchmarks/coreml/python-source-results.json).

## 20-second sample

The Swift figures are medians of three separate processes. Python inference is
the median of four valid runs; its model-plus-inference figure is the median of
the corresponding sums. All four transcripts match byte-for-byte.

| Runtime | Model load (s) | Inference (s) | Inference speed | Startup + inference (s) |
| --- | ---: | ---: | ---: | ---: |
| Python/PyTorch source promoted to FP32 | 2.44 | 0.351 | 57.0x | 2.77 |
| Swift/MLX source BF16 | 0.335 | 0.137 | 145.5x | 0.472 total |
| Swift/MLX converted FP16 | 0.330 | 0.130 | 153.9x | 0.467 total |
| Swift/MLX converted FP32 | 0.356 | 0.115 | 174.2x | 0.477 total |

## Full lecture: same 16 kHz mono WAV

The recording is 6,118.72 seconds (101 minutes 58.72 seconds), rather than one
hour. FP16 is the median of three processes; the other full-length rows are one
process each. Process wall time includes input loading, model loading,
inference, and output handling.

| Runtime | Model load (s) | Inference (s) | Inference speed | Process wall (s) |
| --- | ---: | ---: | ---: | ---: |
| Python/PyTorch source promoted to FP32 | 3.821 | 39.980 | 153.0x | ~46.22 |
| Swift/MLX source BF16 | 0.411 | 26.883 | 227.6x | 27.45 |
| Swift/MLX converted FP16 | 0.376 | 18.937 | 323.1x | 19.76 |
| Swift/MLX converted FP32 | 0.626 | 31.812 | 192.3x | 33.90 |

All three Swift variants produced byte-identical transcripts for this WAV. The
Swift transcript differs from Python by 37 word edits over 13,589 Python words:
0.272% disagreement, BLEU 99.464, chrF2 99.915, and 99.809% character
similarity. This is cross-runtime disagreement, not word error rate; neither
output is ground truth.

## Full lecture: direct WebM input in Swift

This measures the CLI's video-input path, including ffmpeg decoding and
resampling to mono 16 kHz float samples.

| Swift checkpoint | Audio extraction (s) | Model load (s) | Inference (s) | Internal total (s) | Process wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source BF16 | 10.807 | 0.588 | 48.956 | 60.353 | 60.45 |
| Converted FP16 (median of 3) | 10.628 | 0.468 | 29.459 | 40.615 | 42.27 |
| Converted FP32 | 10.530 | 0.354 | 28.528 | 39.417 | 40.78 |

All three Swift variants again produced byte-identical transcripts. Compared
with Python's transcript from the 16-bit WAV, the direct-WebM Swift output has
90 word edits (0.662% disagreement), BLEU 98.741, chrF2 99.576, and 99.538%
character similarity. The WAV and direct-WebM results should not be mixed as a
strict runtime comparison: the decoded sample streams differ because the WAV
was quantized to signed 16-bit PCM, while direct video decoding supplies float
samples.

The macOS full-run peak resident set was approximately 2.2–3.5 GB depending on
checkpoint precision. `time` also reported a roughly 45 GB peak memory
footprint for one-pass inference, reflecting MLX/Metal allocation accounting;
chunked inference remains important for lower-memory Macs even though one-pass
inference works on this 64 GB machine.

## Current bounded Q8 release profile

The current MLX default uses 122.88 seconds of central audio, 10.24 seconds of
context on each side, FP16 activations, and a 64 MiB MLX cache limit. In a
three-round rotated comparison it reduced median speech inference from 25.281
to 21.952 seconds and peak physical footprint from 1.652 to 1.627 GB versus
the previous 20.48-second-context profile.

The selected bounded transcript has 30 word edits against the matching Q8
FP16 unchunked transcript: 99.7797% word agreement and 99.8598% character
similarity across 13,615 unchunked reference words. The previously reported 13
edits compare the new profile with the previous **chunked** profile, not with
one-pass output. These are implementation-agreement diagnostics, not WER.

A manual review found no missing sentences, duplicated passages, or
topic-level corruption. The differences are isolated article, filler, and
single-word alternatives. Several are improvements (`netfli` → `netflix`,
`i is` → `i guess`), while two technical terms differ from one-pass output
(`dk` → `dek` and `dot` → `start`). Only `dot` → `start` was introduced by
the context reduction; the prior bounded profile already emitted `dek`.
