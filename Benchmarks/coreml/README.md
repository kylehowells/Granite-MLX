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
| MLX release Q8 | 24.41 s | 24.88 s | 1.64 GB | 0.169% vs its one-pass output |
| **Core ML Q8/G1, 16,384** | **17.32 s** | **22.09 s** | 2.10 GB | 0.213% vs Core ML FP16 |

Core ML was 29.0% faster for inference and 11.2% faster process-to-process,
while increasing measured peak footprint by about 456 MB. Its raw transcript
had 0.265% word disagreement with the bounded MLX transcript and 99.838%
character similarity. These are reference-output diagnostics, not WER against
a human transcript.

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
