# Granite-MLX

Native Swift/MLX Granite Speech 5.0 TurboCTC CLI for Apple Silicon.

## Native Swift CLI

The Swift runtime now performs end-to-end Granite 5.0 CTC transcription. It
accepts WAV/audio files directly and falls back to `ffmpeg` for video and less
common containers.

```bash
swift build -c release
Scripts/build_mlx_metallib.sh release

.build/release/granite-mlx input.webm
```

`--model` accepts either a Hugging Face repository ID or a local directory.
The default is the published Apache 2.0
[`iky1e/granite-speech-5.0-470m-turboctc-mlx-q8`](https://huggingface.co/iky1e/granite-speech-5.0-470m-turboctc-mlx-q8)
checkpoint, which is downloaded and cached automatically on first use.
The loader can run the original Granite checkpoint directly and also supports
checkpoints produced by `Scripts/convert_granite.py`, including FP16, FP32,
and affine Q8/Q6/Q5/Q4/Q3/Q2 weights.

FP16, Q8, Q6, Q5, and Q4 MLX checkpoints for both the Apache 2.0 and
non-commercial model families are grouped in the
[Granite Speech 5.0 TurboCTC MLX collection](https://huggingface.co/collections/iky1e/granite-speech-50-turboctc-mlx-6a8f4af8ec5acba58088fa38).
The benchmark data used by their comparison tables is in
[`Benchmarks/model-publication`](Benchmarks/model-publication).

Granite's CTC output is lowercase and generally has no sentence punctuation.
The CLI now restores punctuation, capitalization, and sentence boundaries by
default with the native Swift/MLX Q8 formatter
[`iky1e/punctuation-fullstop-truecase-english-mlx-q8`](https://huggingface.co/iky1e/punctuation-fullstop-truecase-english-mlx-q8).
It is downloaded and cached independently on first use. Select another local
or Hugging Face formatter with `--punctuation-model`, or request the exact raw
CTC transcript with `--no-punctuate`:

```bash
granite-mlx lecture.mp4
granite-mlx lecture.mp4 --no-punctuate
granite-mlx lecture.mp4 \
  --punctuation-model iky1e/punctuation-fullstop-truecase-english-mlx-fp16
```

The native output modes match the Parakeet MLX CLI: `txt`, `srt`, `vtt`,
`json`, and `all`. SRT is the default. Granite's approximate CTC word times
and the formatter's predicted sentence boundaries drive subtitle cues:

```bash
granite-mlx lecture.mp4 --output-format vtt
granite-mlx lecture.mp4 --output-format all --output-dir ./transcripts
granite-mlx lecture.mp4 interview.m4a \
  --output-dir ./transcripts \
  --output-template '{filename}-{date}-{index}'
granite-mlx lecture.mp4 --output-format srt --highlight-words
```

Generated paths never overwrite existing files; a numeric suffix is added on
collision. For multiple inputs without `--output-dir`, stdout is JSON Lines so
each result remains distinct. JSON preserves `raw_text`, user-facing `text`,
optional `formatted_text`, timed words and subtitle segments, model settings,
and performance metadata. All progress and benchmark records go to stderr.

### Model downloads and disk usage

The first transcription automatically downloads the selected speech model and,
unless `--no-punctuate` is used, the punctuation model. The CLI reports the
repository, approximate size, cache destination, completion percentage, and
transfer speed on stderr. Subsequent runs use the local cache without network
access or progress noise.

Use the built-in model manager to inspect all 15 published checkpoints and the
space occupied by downloaded models:

```bash
granite-mlx models list
granite-mlx models list --downloaded-only
granite-mlx models list --json
```

Models can be selected by the short aliases shown in that list:

```bash
granite-mlx models download apache-q8 punctuation-q8
granite-mlx models download nc-q6
granite-mlx lecture.mp4 --model apache-q8
```

Remove one model, or every Granite-compatible model in this cache, to reclaim
space. Removal asks for confirmation unless `--yes` is supplied and never
targets unrelated models that happen to share the Swift Hugging Face cache:

```bash
granite-mlx models remove apache-fp16
granite-mlx models remove punctuation-q4 --yes
granite-mlx models remove --all
```

The materialized cache is under `~/Documents/huggingface/models`. Removed
models are permanently deleted from the local cache but can be downloaded
again at any time. `--hf-token` is available for private or gated repository
IDs; when omitted, the Swift Hub library can use its normal environment token.

The published formatter FP16/Q8/Q6/Q5/Q4 checkpoints are grouped in the
[Granite Speech 5.0 — Punctuation & Capitalization Model MLX collection](https://huggingface.co/collections/iky1e/granite-speech-50-punctuation-and-capitalization-model-mlx-6a8f6ad45d0f10d3f0bbc5b2).
Q8 is the default: its 55.8 MB weight file retained 99.9536% character
agreement with the original ONNX FP32 formatter on the full 101m59s lecture
transcript. See [`Benchmarks/punctuation`](Benchmarks/punctuation).

The CLI defaults to FP16 encoder activations, with numerically sensitive
softmax work kept in FP32. Use `--activation-precision baseline` to reproduce
the original FP32 activation path. `fp8-emulated` and `int8-emulated` are
diagnostic storage experiments: this M1 Max/MLX stack has no native FP8 or INT8
activation matrix-multiplication path, and both modes are slower than FP16.

The CLI defaults to a bounded-memory profile while retaining a single global
CTC collapse and tokenizer decode:

```bash
granite-mlx lecture.wav \
  --model /path/to/granite-speech-5.0-470m-turboctc-mlx-q8-g128 \
  --mlx-cache-limit-mb 64 \
  --audio-chunk-duration 122.88 \
  --audio-chunk-context 20.48
```

The chunk and context durations are aligned to Granite's 10.24-second attention
blocks. On the 101m59s test lecture this profile reduced macOS peak footprint
from 34.67 GB to 1.64 GB, ran in 24.41 seconds, and had 0.169% word disagreement
against the same Q8 one-pass transcript. These comparisons are diagnostics,
not ground-truth WER. See
[`Benchmarks/memory-optimization`](Benchmarks/memory-optimization).

Run the complete recording through the encoder in one pass with:

```bash
granite-mlx lecture.wav --no-chunking
```

`--audio-chunk-duration 0` provides the same no-chunking behavior for scripts.
The 64 MiB MLX cache limit remains active unless another value is supplied.

For exact one-pass output with less retained memory, use
`--mlx-cache-limit-mb 0 --middle-ctc-vocabulary-tile 2048`. This reduced the
same run to 9.66 GB without changing the transcript, but temporal chunking is
still required for mobile-scale memory.

Create a quantized checkpoint with FP16 residual tensors:

```bash
uv run python Scripts/convert_granite.py \
  /path/to/granite-speech-5.0-470m-turboctc \
  /path/to/granite-speech-5.0-470m-turboctc-mlx-q4 \
  --precision fp16 \
  --quantization-bits 4 \
  --group-size 64
```

The converter quantizes all 2-D Linear and relative-position Embedding weights.
Convolution, normalization, and bias tensors remain FP16. The generated
`config.json` records the quantization mode, bit width, and group size; the
Swift loader detects it automatically and validates every packed weight,
scale, and quantization-bias shape before inference.

Q8 through Q4 remain useful on the current validation lecture. Q3 has visible
transcription degradation and Q2 collapses to an empty transcript, so neither
should be distributed as a quality preset. See
[`Benchmarks/quantization`](Benchmarks/quantization) for raw JSON, transcripts,
methodology, and PNG charts.

For repeatable performance runs, use `--benchmark`. The selected output remains
on stdout and one machine-readable timing record is written to stderr:

```bash
granite-mlx input.webm \
  --model /path/to/granite-speech-5.0-470m-turboctc-mlx-fp16 \
  --benchmark > transcript.srt 2> benchmark.jsonl
```

The timing record includes audio/model load time, inference time, punctuation
model/load/inference time, real-time factor, real-time multiple, activation
precision, and CTC vocabulary tile size.
Run once as a Metal warm-up before recording steady-state performance.

The recommended Q8 checkpoint uses group size 128 where the matrix shape
allows it and group size 64 for `encoder.input_linear`. On the paired full
lecture benchmark it is about 2% faster and 14 MiB smaller than uniform G64,
with the same 0.066% transcript disagreement relative to the FP32-weight Swift
reference. See [`Benchmarks/q8-optimization`](Benchmarks/q8-optimization) for
the reproducible JSON, Metal trace exports, microbenchmarks, and PNG charts.

Reproduce the complete quantization benchmark with:

```bash
uv run python Scripts/benchmark_quantization.py \
  --binary /path/to/release/granite-mlx \
  --model-root /path/to/ML-Models \
  --short-audio /path/to/20s.wav \
  --full-audio /path/to/lecture-16k-mono.wav \
  --python-short-reference /path/to/python-20s.txt \
  --python-full-reference-json /path/to/python-full.json \
  --output Benchmarks/quantization/results.json

uv run python Scripts/chart_quantization.py \
  Benchmarks/quantization/results.json \
  Benchmarks/quantization/charts
```

## Reference CLI

The Python reference CLI defines the command-line behavior and provides the
PyTorch baseline for validating the Swift implementation. It is development
tooling only and will not be required by the finished native CLI.

Install its dependencies in an isolated environment, then run:

```bash
python -m pip install -e .
granite-reference /path/to/audio.wav \
  --model /Users/kylehowells/Developer/ML-Models/granite-speech-5.0-470m-turboctc \
  --output-format srt \
  --verbose
```

The output modes match Parakeet MLX: `txt`, `srt`, `vtt`, `json`, and `all`.
SRT/VTT produce readable timestamped phrase cues. Add `--highlight-words` to
produce one cue per aligned word with the active word underlined in SRT or
bolded in VTT. JSON includes the transcript, model/runtime metadata, CTC
token timings, approximate word timings, and subtitle segments.

Granite is CTC rather than TDT, so its word timings are derived from emission
frames and are approximate. They are suitable for subtitle navigation and
highlighting, but should not be treated as frame-accurate forced alignment.

Generate every output format at once:

```bash
granite-reference /path/to/audio.wav \
  --model /Users/kylehowells/Developer/ML-Models/granite-speech-5.0-470m-turboctc \
  --output-dir ./transcripts \
  --output-format all \
  --highlight-words
```

The native transcript has been checked character-for-character against this
reference on the fixed 20-second fixture. The native CTC timing, subtitle
segmentation, all five exporters, highlighted-word mode, output templates,
collision handling, and batch model reuse are now implemented in Swift.
