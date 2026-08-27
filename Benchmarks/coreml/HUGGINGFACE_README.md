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

| Runtime | Model artifact | Speech path | Process wall | Peak footprint | Agreement with native Swift source output |
|---|---:|---:|---:|---:|---:|
| IBM source / Python PyTorch FP32, one pass | 902.35 MiB | 25.69 s | 30.47 s | 27.48 GB | 97.5395% |
| IBM source / native Swift, bounded | 902.35 MiB | 29.77 s | 30.42 s | 2.61 GB | 100.0000% |
| Granite-MLX Q8 | 466.03 MiB | 24.41 s | 24.88 s | 1.64 GB | 99.8825% |
| **Core ML Q8 (this repository)** | **659.84 MiB** | **17.32 s** | **22.09 s** | **2.10 GB** | **99.7797%** |

Agreement is `100 − Levenshtein word edits / native-Swift source-output words`;
it is not WER and does not measure correctness against a human transcript.
Backend precision and one-pass versus bounded long-form decoding can both
change output. The Core ML transcript had 30 word edits across 13,615 reference
words and 99.870% character similarity. The formatting-enabled Granite CLI
path took 19.83 seconds of processing and retained 99.65% lexical agreement
with the formatted MLX result.

The Python source-weight row is a fresh median of three clean processes using
IBM's documented `AutoModelForCTC.generate()` path, Transformers 5.16.1,
PyTorch 2.13.0, and MPS. Its speech path comprises 0.464 seconds of frontend
processing, 25.195 seconds in `model.generate()`, and 0.031 seconds of tokenizer
decode. Median model load was 0.594 seconds and audio load was 0.223 seconds.
Maximum RSS was 1.40 GB, the MPS driver allocator held 26.22 GB, and macOS
reported a 27.48 GB peak physical footprint. These memory views are reported
separately and are not summed. All three FP32 transcripts were byte-identical.

The prescribed BF16 model/input cast was also run three times. It used 14.30 GB
and took 24.64 seconds for the speech path, but on this M1 Max MPS backend it
produced a deterministic, incomplete 2,523-word long-form decode rather than a
complete roughly 13,600-word transcript. It is retained in the raw results as
a compatibility finding and is not used as the valid Python baseline.

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
transcripts, and a PNG comparison chart are maintained in Granite-MLX's
[`Benchmarks/coreml`](https://github.com/kylehowells/Granite-MLX/tree/master/Benchmarks/coreml)
directory. This repository also contains the complete Python source-weight
[`benchmark JSON`](benchmarks/python-source-results.json) and its deterministic
[`FP32`](benchmarks/python-source-fp32.txt) and
[`BF16`](benchmarks/python-source-bf16.txt) transcripts.

## Files

- `GraniteSpeech.mlpackage` — fixed-shape Q8 Core ML ML Program.
- `coreml_config.json` — machine-readable package, shape, precision, and
  recommended-runtime metadata.
- `config.json` and tokenizer/processor JSON files — matching Granite
  architecture and CTC tokenizer metadata.
- `benchmarks/` — fresh Python source-weight raw-run summaries and transcripts.

## License

These converted **model weights** retain the source model's Apache 2.0 terms.
The Granite-MLX Swift software is separately dual-licensed under MIT or Apache
2.0; installing the software does not bundle these model weights.
