---
license: apache-2.0
language:
- en
pipeline_tag: automatic-speech-recognition
library_name: coreml
base_model: ibm-granite/granite-speech-5.0-470m-turboctc
tags:
- coreml
- swift
- granite
- speech-to-text
- automatic-speech-recognition
- apple-silicon
---

# Granite Speech 5.0 470M TurboCTC — Core ML Q8

This repository contains an **8-bit Core ML conversion** of IBM's
[`ibm-granite/granite-speech-5.0-470m-turboctc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc)
for native Swift inference with
[`Granite-MLX`](https://github.com/kylehowells/Granite-MLX).

It was converted from source revision
[`7e74c6438b7cfb5090cb6a131538f5e8515a7de3`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc/tree/7e74c6438b7cfb5090cb6a131538f5e8515a7de3).
The tokenizer, processor metadata, and Granite configuration in this repository
come from that source checkpoint. The original BF16 weights are not included.

## Model configuration

| Property | Value |
|---|---:|
| Core ML format | ML Program (`.mlpackage`) |
| Minimum deployment target | macOS 15.0 |
| Fixed input | `[1, 16384, 320]` FP16 frontend features |
| Audio represented per invocation | 327.68 seconds |
| Output | `[1, 4096]` greedy CTC token IDs (`Int32`) |
| Weight compression | Uniform 8-bit palettization |
| Quantization granularity | Per grouped channel, group size 1 |
| Recommended compute units | CPU + GPU |
| Package size | 659.84 MiB |
| Weight SHA-256 | `3ef27cdaa3968451f92c74d33c9b8cd885b3793a751c16d19acdc0fb8f07eaf4` |

Granite-MLX reserves 20.48 seconds of context on each side and advances by
286.72 seconds when transcribing long recordings. It performs one global CTC
collapse and tokenizer decode after all central frame emissions are joined.

## Benchmark and output agreement

The benchmark input is a 6,118.72-second (101m58.72s) predominantly
single-speaker Stanford CME295 lecture, decoded to mono 16 kHz audio. Times are
from an Apple M1 Max with 64 GB unified memory on macOS 26.5.2.

| Runtime | Model artifact | Speech inference, median (min–max) | Process wall, median | Peak footprint | Agreement with native Swift source output |
|---|---:|---:|---:|---:|---:|
| IBM source BF16 / Python MPS, one pass | 902.35 MiB | 33.78 s (31.50–34.99) | 39.06 s | 14.30 GB | 97.5835% |
| IBM source cast to FP16 / Python MPS, one pass | 902.35 MiB | 24.34 s (24.01–26.30) | 29.00 s | 14.34 GB | 97.5468% |
| IBM source promoted to FP32 / Python MPS, one pass | 902.35 MiB | 30.95 s (30.42–31.18) | 35.96 s | 27.48 GB | 97.5395% |
| IBM source / native Swift MLX, bounded | 902.35 MiB | 46.88 s (46.29–51.73) | 47.33 s | 2.67 GB | 100.0000% |
| Converted FP16 / native Swift MLX, one pass | 902.22 MiB | **21.28 s (21.16–22.99)** | **21.94 s** | 13.25 GB | 99.7650% |
| Granite-MLX Q8, bounded | 466.03 MiB | 37.11 s (36.35–37.17) | 37.62 s | **1.72 GB** | 99.8825% |
| **Core ML Q8 (this repository), bounded** | **659.84 MiB** | **24.50 s (24.32–26.38)** | **26.16 s** | **2.04 GB** | **99.7797%** |

Agreement is `100 − Levenshtein word edits / native-Swift source-output words`;
it is not WER and does not measure correctness against a human transcript.
Backend precision and one-pass versus bounded long-form decoding can both
change output. The Core ML transcript had 30 word edits across 13,615 reference
words and 99.870% character similarity.

These are three-run medians from complete interleaved rounds, not three
consecutive runs of each model. Every configuration produced a byte-identical
transcript across rounds. Speech-time coefficient of variation ranged from
1.23% to 6.18%; the first round was usually slower and rounds two and three
converged.

![Three-round Granite backend benchmark](coreml-vs-mlx.png)

The Python source-weight rows use
`AutoModelForCTC.generate()`, Transformers 5.16.1, PyTorch 2.13.0, and MPS.
The checkpoint stores BF16 weights. Runtime FP16 casts those BF16 values to
FP16; runtime FP32 merely promotes them and does not recover extra information.

FP16 remained the fastest complete Python path at 24.34 seconds median speech
time. It differed from promoted FP32 by one word across 13,595 words (99.9926%
agreement) while reducing peak physical footprint from 27.48 GB to 14.34 GB.
Native BF16 took 33.78 seconds at 14.30 GB and retained 99.9117% agreement with
promoted FP32.

There is an important MPS workaround in these valid 16-bit rows. Pre-casting
the full `[1, 305936, 320]` feature tensor as in the current documented example,
then releasing its 391.6 MB FP32 source before `generate()`, silently corrupted
both BF16 and FP16 long-form output. Retaining that source tensor or deferring
the cast to Granite's first layer produced complete deterministic transcripts.
The behavior matches PyTorch's known allocator-state-dependent MPS wrong-result
defect ([#193487](https://github.com/pytorch/pytorch/issues/193487)) and its
related greater-than-65,536-row biased-linear defect
([#189495](https://github.com/pytorch/pytorch/issues/189495)). Both corrupted
paths remain preserved in the benchmark JSON rather than being presented as
valid accuracy results.

With the user-facing Q8 formatter enabled, Core ML took 24.60 seconds for
speech, 1.10 seconds for punctuation, 25.78 seconds processing total, and 26.59
seconds process wall. The formatted transcript retained 99.7140% lexical word
agreement and 99.4544% character similarity with formatted MLX output.

Core ML's CPU+GPU policy was substantially faster than CPU+Neural Engine on the
tested M1 Max. Granite contains 32 dynamic attention matmuls that the M1 Neural
Engine compiler could not place, so CPU/ANE graph partitioning was slower.

## Usage

Install a Granite-MLX release containing Core ML model-catalog support, then:

```shell
granite-mlx recording.m4a --backend coreml
```

The CLI downloads this model and the default punctuation model automatically.
It accepts common audio and video files and exports TXT, SRT, WebVTT, JSON, or
all formats. Explicit model management is also available:

```shell
granite-mlx models list
granite-mlx models download apache-coreml-q8
granite-mlx models remove apache-coreml-q8
```

Native library clients can use `GraniteCoreMLRecognizer` with this repository
ID and receive download progress, cancellation, approximate CTC word timings,
and structured diagnostic errors.

## Reproducing the conversion

From the Granite-MLX repository at commit `922f577`:

```shell
uv sync

uv run python Scripts/convert_granite_coreml.py \
  /path/to/granite-speech-5.0-470m-turboctc \
  /path/to/granite-coreml-fp16-16384.mlpackage \
  --feature-frames 16384 --minimum-target macos15

uv run python Scripts/quantize_granite_coreml.py \
  /path/to/granite-coreml-fp16-16384.mlpackage \
  /path/to/GraniteSpeech.mlpackage \
  --method palette-uniform --bits 8 \
  --granularity per-channel --block-size 1
```

Converter SHA-256:

- `convert_granite_coreml.py`: `35ca20e295de822d1d93ba853f05b53c8c0991f191836e8ec1535f2de98e8b55`
- `quantize_granite_coreml.py`: `2035c8c39989ba65d8b50c4377f3c1bd9644e8d0b4a2f09b12da6d3541f4d515`

Detailed conversion experiments, negative ANE/INT8 findings, complete JSON,
transcripts, and the PNG comparison chart are maintained in Granite-MLX's
[`Benchmarks/coreml`](https://github.com/kylehowells/Granite-MLX/tree/master/Benchmarks/coreml)
directory. This repository contains the current interleaved
[backend matrix](benchmarks/backend-matrix-results.json), its deterministic
transcripts, and the earlier complete Python source-weight
[`benchmark JSON`](benchmarks/python-source-results.json) and its deterministic
[`BF16`](benchmarks/python-source-bf16.txt),
[`FP16`](benchmarks/python-source-fp16.txt), and
[`FP32`](benchmarks/python-source-fp32.txt) transcripts, plus the two pre-cast
failure-mode transcripts.

## Files

- `GraniteSpeech.mlpackage` — fixed-shape Q8 Core ML ML Program.
- `coreml_config.json` — machine-readable package, shape, precision, and
  recommended-runtime metadata.
- `config.json` and tokenizer/processor JSON files — matching Granite
  architecture and CTC tokenizer metadata.
- `coreml-vs-mlx.png` — median and min–max chart for the current three-round matrix.
- `benchmarks/backend-matrix-results.json` — all 27 current run summaries,
  stability statistics, memory counters, and comparisons.
- `benchmarks/backend-matrix-*.txt` — deterministic current transcripts.
- `benchmarks/python-*` — earlier Python MPS diagnostics, including invalid
  pre-cast failure modes.

## License

These converted **model weights** retain the source model's Apache 2.0 terms.
The Granite-MLX Swift software is separately dual-licensed under MIT or Apache
2.0; installing the software does not bundle these model weights.
