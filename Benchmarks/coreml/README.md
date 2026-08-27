# Core ML backend experiments

These experiments converted Granite Speech 5.0 TurboCTC into a fixed-shape
Core ML ML Program and searched compute-unit placement, graph output,
quantization, and long-audio chunk sizing on an M1 Max Mac. The result is a
native Swift Core ML backend that can beat this project's release MLX backend.
The selected package is published with its tokenizer, configuration, source
provenance, checksums, and model card at
[`iky1e/granite-speech-5.0-470m-turboctc-coreml-q8`](https://huggingface.co/iky1e/granite-speech-5.0-470m-turboctc-coreml-q8).

## Result

The best configuration is uniform Q8 palettization with one scale/palette per
output channel (`group_size=1`), a macOS 15 deployment target, CPU+GPU compute,
and a 16,384-frame input. The graph represents 327.68 seconds of audio; the CLI
keeps 20.48 seconds of context on each side and advances by 286.72 seconds.

| Backend | Speech inference | Process wall | Peak footprint | Raw comparison |
|---|---:|---:|---:|---:|
| IBM source BF16 / Python MPS, one pass | 23.65 s | 28.91 s | 14.30 GB | 2.416% vs native Swift source output |
| IBM source cast to FP16 / Python MPS, one pass | 17.36 s | 21.66 s | 14.34 GB | 2.453% vs native Swift source output |
| IBM source promoted to FP32 / Python MPS, one pass | 25.69 s | 30.47 s | 27.48 GB | 2.461% vs native Swift source output |
| MLX release Q8 | 24.41 s | 24.88 s | 1.64 GB | 0.169% vs its one-pass output |
| **Core ML Q8/G1, 16,384** | **17.32 s** | **22.09 s** | 2.10 GB | 0.213% vs Core ML FP16 |

Core ML was 29.0% faster for inference and 11.2% faster process-to-process,
while increasing measured peak footprint by about 456 MB. Its raw transcript
had 0.265% word disagreement with the bounded MLX transcript and 99.838%
character similarity. These are reference-output diagnostics, not WER against
a human transcript.

The Python rows are fresh three-run medians through
`AutoModelForCTC.generate()` using Transformers 5.16.1 and PyTorch 2.13.0 on
MPS. The checkpoint is stored as BF16. “FP16” casts those BF16 values to FP16
at runtime; “FP32” promotes the same values and cannot restore information that
was not present in the checkpoint.

FP16 was the fastest valid Python path: 0.424 seconds of frontend processing,
16.903 seconds in `model.generate()`, 0.029 seconds of tokenizer decode, and
17.359 seconds total. It emitted 13,595 words, differed from promoted FP32 by
one word (99.9926% agreement), and used a 14.34 GB peak physical footprint.
Native BF16 took 23.65 seconds, used 14.30 GB, and retained 99.9117% agreement
with promoted FP32. The native-Swift comparison additionally includes MPS
versus MLX numerical behavior and one-pass versus bounded long-form decoding;
it is not model accuracy or human-reference WER.

The current documented input preparation pre-casts the full feature tensor
before `generate()`. For this recording that tensor is `[1, 305936, 320]`. On
PyTorch 2.13 MPS, releasing its 391.6 MB FP32 source before inference silently
corrupted both BF16 and FP16 long-form results. Either retaining that source or
leaving features FP32 until Granite's first layer performs its mandatory cast
produced complete, deterministic output. This matches the known class of
allocator-state-dependent MPS wrong-result defects in
[`pytorch/pytorch#193487`](https://github.com/pytorch/pytorch/issues/193487)
and the related greater-than-65,536-row biased-linear defect in
[`pytorch/pytorch#189495`](https://github.com/pytorch/pytorch/issues/189495).
Complete valid and failure-mode runs, memory counters, environment metadata,
and transcripts are in `python-source-results.json` and the adjacent TXT files.

A same-session paired comparison also ran converted Swift MLX FP16 and Python
MPS FP16 with chunking disabled. Swift's three-run median was 22.07 seconds
versus Python's 24.87 seconds, making Swift 11.3% faster under the same sustained
load. The result confirms that repeated bounded chunks, rather than Swift
language overhead, explain the earlier apparent gap. Absolute one-pass timing
was highly session-sensitive, and Swift's 13.25 GB peak remains unsuitable as
the production default. Full runs are in
[`../no-chunk-comparison`](../no-chunk-comparison).

With the user-facing Q8 punctuation model enabled, Core ML took 18.65 seconds
for speech and 1.05 seconds for formatting. Processing finished in 19.83
seconds and process wall time was 24.40 seconds. The formatted transcript had
99.65% lexical word agreement with the formatted MLX result. Punctuation and
case make strict formatted word edits look larger (2.25%) even when lexical
content is unchanged, so both measurements are preserved in `results.json`.

![Core ML versus MLX benchmark](coreml-vs-mlx.png)

## Why the Neural Engine did not win

The Core ML compute plan prefers the Neural Engine for 722 operations, but 32
dynamic attention matmuls are unsupported: query-key and attention-value
matmuls in every Conformer layer. Core ML therefore partitions the graph
between CPU and ANE. On this M1 Max, CPU+ANE took about 103 ms for the short
fixed graph versus 53 ms for CPU+GPU. `computeUnits=all` behaved similarly.

This does not prove that every newer Apple chip behaves identically. It does
show that simply selecting the ANE is not an optimization for this architecture
on M1; the unsupported dynamic attention operations and transfer/partition
cost dominate any ANE gains.

## Quantization findings

- Core ML linear INT8 weight quantization was approximately 14x slower than
  FP16 on the 8,192-frame graph despite preserving the short transcript.
- Per-tensor Q8/Q6/Q4 palettes were fast but destroyed recognition accuracy.
- Grouped-channel Q8 palettes were fast. Group 32 reduced package size to 487
  MiB but caused 0.977% word disagreement on the lecture. Group 1 retained the
  best accuracy at 660 MiB and is the recommended choice.
- Q6/G1 is 406 MiB but caused 0.507% disagreement, a 4.74 GB measured peak, and
  lazy-compilation latency. It is not a better release default.
- Q4/G1 retained the short transcript despite 1.56% frame-ID disagreement, but
  was not promoted to a full-audio candidate.
- Core ML supports 1/2/3/4/6/8-bit palettes, not 5-bit palettes.
- INT8 activation calibration completed with a coarse calibration group but
  the resulting graph crashed in Core ML. Core ML Tools 9 offered no usable
  FP8 activation conversion for this graph.

The 16,384-frame graph is the speed/memory sweet spot measured here. 32,768
frames was slightly slower at 18.32 seconds and used 2.88 GB; 24,576 frames was
both slower and thermally inconsistent. The CLI now automatically fills the
selected Core ML graph after reserving its context, so users do not need the
benchmark's explicit chunk-size arguments.

## Reproduce

Install the locked Python environment, convert FP16, then palettize it:

```bash
uv sync
uv run python Scripts/convert_granite_coreml.py \
  /path/to/granite-speech-5.0-470m-turboctc \
  /path/to/granite-coreml-fp16-16384.mlpackage \
  --feature-frames 16384 --minimum-target macos15

uv run python Scripts/quantize_granite_coreml.py \
  /path/to/granite-coreml-fp16-16384.mlpackage \
  /path/to/granite-coreml-q8-g1-16384.mlpackage \
  --method palette-uniform --bits 8 \
  --granularity per-channel --block-size 1
```

Build and run the native backend. The published package is downloaded and
cached automatically:

```bash
swift build -c release
Scripts/build_mlx_metallib.sh release

.build/release/granite-mlx lecture.wav \
  --backend coreml \
  --output-format txt --benchmark
```

For a local conversion, additionally pass
`--coreml-model /path/to/model.mlpackage --model /path/to/tokenizer`.

The first load downloads the repository and compiles the package into an
OS-specific `.mlmodelc`
under `~/Library/Caches/GraniteMLX/CoreML`. Later runs reuse it. Core ML can
still perform lazy device preparation on model load, so model-load timing is
reported separately from speech inference.

`results.json` contains every number summarized above. Generate the checked
PNG with `uv run python Scripts/chart_coreml.py`.
