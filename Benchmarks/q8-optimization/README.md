# Q8-native optimization results

> These optimization measurements deliberately use one-pass inference to
> isolate kernel and precision changes. The production CLI instead defaults to
> bounded temporal chunks; see [`../memory-optimization`](../memory-optimization)
> for representative current memory use.

The recommended Granite-MLX configuration is affine Q8 with group size 128
where matrix dimensions permit it, group size 64 for
`encoder.input_linear`, and FP16 encoder activations with FP32 softmax
stability islands.

On the 64 GB M1 Max, three interleaved one-pass runs of the fixed 6,118.72
second lecture produced these medians:

| Configuration | Inference | Throughput | Checkpoint | Word disagreement vs Swift FP32 |
|---|---:|---:|---:|---:|
| Q8 G64 + FP16 activations | 20.68 s | 295.9x | 480.09 MiB | 0.0661% |
| Mixed Q8 G128/G64 + FP16 activations | 20.25 s | 302.1x | 466.03 MiB | 0.0661% |

The transcript comparison is a diagnostic against the FP32-weight Swift
output, not ground-truth WER. Both Q8 configurations were deterministic across
their three runs.

## What improved performance

The Metal trace showed the original Q8 runtime feeding FP32 frontend values to
the generic affine QMM kernel. Casting the encoder path to FP16 was the largest
successful optimization. Increasing compatible weight groups from 64 to 128
then reduced scale/bias traffic and consistently improved adjacent interleaved
runs by roughly 1.2--2.1%.

## Experiments not selected

- A numerically valid symmetric signed-Q8/G128 custom kernel was 2x slower for
  short frame counts and about 9--13.5x slower for long frame counts than MLX's
  stock affine QMM. It was rejected.
- An experimental specialized BM64 MLX QMM tile failed transcript correctness
  and did not improve speed. The dependency and metallib were restored to the
  stock implementation.
- Exact vocabulary-tiled CTC projection with a running argmax was implemented
  using public MLX operations. Tile 2048 preserved the transcript exactly but
  was 1.40% slower and did not lower peak memory because the middle CTC logits
  establish the peak. True fusion requires a new QMM primitive/Metal epilogue
  that emits maxima and token IDs without materializing logits.
- This M1 Max with MLX Python 0.31.2 / MLX Swift 0.31.4 cannot use ordinary
  integer matmul for INT8 activations, and its FP8 QQMM path reports the common
  Granite shapes as unsupported. FP8 E4M3 and signed INT8 were therefore tested
  as byte storage between layers with FP16 compute. Against adjacent FP16
  controls, FP8 was 14.01% slower with 0.264% word disagreement; INT8 was 5.76%
  slower with 1.043% disagreement. Neither lowered full-model peak memory.

## Artifacts

- `results.json` is the consolidated source of truth, including raw run values,
  paired controls, file sizes, accuracy diagnostics, and capability results.
- `qmm-results.json` contains the per-shape/per-frame QMM microbenchmarks and
  low-precision capability probes.
- `dtype-audit-fp16.json` records evaluated activation shapes, dtypes, ranges,
  and byte counts.
- `metal/trace-toc.xml` and `metal/shaders.xml` preserve the exported Metal
  trace metadata and shader list. The 36 MB raw trace remains outside Git.
- `charts/*.png` are generated directly from `results.json` by
  `Scripts/chart_q8_optimization.py`.

Timing on this machine varied with surrounding system load, so experimental
modes are compared with immediately adjacent FP16 controls. G64 and G128 use
three interleaved runs. Absolute timings from different run sessions should not
be compared directly.

## Parakeet-MLX reference

The installed `parakeet-mlx` 0.2.4 CLI was also measured on the identical WAV
using its default `mlx-community/parakeet-tdt-0.6b-v2` BF16 model and production
120-second chunks with 15 seconds of overlap. With the model loaded outside the
timed region, three runs were 53.05, 53.61, and 53.31 seconds: a 53.31-second
median, or 114.8x real time. The transcripts were identical across runs.

Granite's mixed-G128 result is 2.63x faster at 20.25 seconds and 302.1x real
time. This is not a perfectly like-for-like architecture comparison: Granite
runs the lecture in one pass, while Parakeet processes 6,988.72 seconds after
chunk overlap (14.2% extra audio). The resulting memory trade-off is large.
Granite's measured macOS peak footprint was about 34.67 GB, while a complete
Parakeet CLI run used about 4.66 GB. Parakeet's unchunked path uses full relative
attention and is not a safe or useful 102-minute benchmark on this machine.

Raw Parakeet timings and the preserved transcript are under `parakeet/`.
