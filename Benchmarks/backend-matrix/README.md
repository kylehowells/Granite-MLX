# Three-round backend benchmark

This benchmark reruns every runtime row used by the Core ML model card, plus
the converted Swift FP16 one-pass path and both formatting-enabled defaults.
Each configuration transcribed the complete 6,118.72-second (101m58.72s)
Stanford CME295 lecture three times on an Apple M1 Max with 64 GB unified
memory. The order was interleaved and repeated from start to finish; no
configuration ran three times consecutively.

The table reports the median of three fresh processes. Parentheses show the
minimum and maximum speech inference time. Peak memory is macOS peak physical
footprint.

| Runtime | Artifact | Speech inference, median (min–max) | Process wall, median | Peak footprint | Agreement with native Swift source |
|---|---:|---:|---:|---:|---:|
| IBM source BF16 / Python MPS, one pass | 902.35 MiB | 33.78 s (31.50–34.99) | 39.06 s | 14.30 GB | 97.5835% |
| IBM source cast to FP16 / Python MPS, one pass | 902.35 MiB | 24.34 s (24.01–26.30) | 29.00 s | 14.34 GB | 97.5468% |
| IBM source promoted to FP32 / Python MPS, one pass | 902.35 MiB | 30.95 s (30.42–31.18) | 35.96 s | 27.48 GB | 97.5395% |
| IBM source / native Swift MLX, bounded | 902.35 MiB | 46.88 s (46.29–51.73) | 47.33 s | 2.67 GB | 100.0000% |
| Converted FP16 / native Swift MLX, one pass | 902.22 MiB | **21.28 s (21.16–22.99)** | **21.94 s** | 13.25 GB | 99.7650% |
| Granite MLX Q8 / native Swift, bounded | 466.03 MiB | 37.11 s (36.35–37.17) | 37.62 s | **1.72 GB** | 99.8825% |
| **Core ML Q8 / native Swift, bounded** | **659.84 MiB** | **24.50 s (24.32–26.38)** | **26.16 s** | **2.04 GB** | **99.7797%** |

The current Core ML Q8 backend is 34.0% faster for speech inference than the
bounded MLX Q8 default and 30.5% faster process-to-process. It uses about
318 MB more peak physical memory. One-pass Swift FP16 is faster than either,
but its 13.25 GB footprint is unsuitable as the low-resource default.

## Formatting-enabled defaults

Formatting uses the Q8 MLX punctuation and true-casing model after speech
recognition.

| Runtime | Speech inference | Formatter inference | Processing total | Process wall | Peak footprint |
|---|---:|---:|---:|---:|---:|
| MLX Q8 + formatter | 35.88 s | 1.09 s | 37.06 s | 37.91 s | 1.76 GB |
| **Core ML Q8 + formatter** | **24.60 s** | **1.10 s** | **25.78 s** | **26.59 s** | **2.11 GB** |

The Core ML formatted transcript has 99.7140% lexical word agreement and
99.4544% character similarity with the formatted MLX transcript. Strict
formatted word agreement is 97.8178% because punctuation and capitalization
are included. These are implementation-agreement diagnostics, not WER against
a human transcript.

## Stability

All nine configurations produced byte-identical transcripts across their
three rounds. Speech-time coefficient of variation ranged from 1.23% to 6.18%.
The first round was usually slower, while rounds two and three converged. The
machine still had moderate background CPU activity, so these measurements are
best treated as a reproducible same-session comparison rather than the fastest
possible M1 Max result.

[results.json](results.json) retains every run, min/max, mean, median,
standard deviation, coefficient of variation, memory counters, machine load
snapshots, transcript hashes, and cross-runtime comparisons. Deterministic
transcripts are under [transcripts](transcripts).

## Reproduce

Build the release executable and run the interleaved matrix:

~~~bash
swift build -c release
Scripts/build_mlx_metallib.sh release

uv run python Scripts/benchmark_backend_matrix.py \
  --binary .build/release/granite-mlx \
  --audio /path/to/lecture-16k-mono.wav \
  --source-model /path/to/granite-speech-5.0-470m-turboctc \
  --mlx-fp16-model /path/to/granite-speech-5.0-470m-turboctc-mlx-fp16 \
  --mlx-q8-model /path/to/granite-speech-5.0-470m-turboctc-mlx-q8-g128 \
  --coreml-model /path/to/GraniteSpeech.mlpackage \
  --punctuation-model /path/to/punctuation-fullstop-truecase-english-mlx-q8 \
  --output /path/to/raw-results \
  --rounds 3

uv run python Scripts/summarize_backend_matrix.py \
  --raw-root /path/to/raw-results \
  --output Benchmarks/backend-matrix

uv run python Scripts/chart_coreml.py
~~~
