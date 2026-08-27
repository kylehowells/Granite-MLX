# Activation Precision Benchmark Plan

## Objective

Measure whether reducing Granite Speech activation precision lowers peak memory
and/or improves inference speed without materially changing transcription output.
The primary candidates are FP16 and FP8 activations, compared with the current
activation behavior as the baseline.

This benchmark concerns runtime activations, not checkpoint weight precision.
Weight-only quantization and activation precision must be recorded separately in
every result.

## Questions to answer

1. What dtype does each major part of the current Swift/MLX runtime actually
   produce?
2. How much of the roughly 44--45 GB one-pass macOS peak memory footprint is
   reduced by FP16 activations?
3. Does FP16 improve speed for Q8, Q6, and Q5 weights, or introduce extra casts
   that offset its memory benefit?
4. Does the pinned MLX Swift version provide a real FP8 activation and compute
   path on the test Mac, or would FP8 require emulation/custom kernels?
5. What transcript disagreement does each activation mode introduce relative to
   the same weight checkpoint using baseline activations?

## Important terminology

- **Weight precision:** checkpoint storage and matrix weight representation,
  such as Q8, Q6, or Q5.
- **Activation precision:** dtype used for hidden states and intermediate
  tensors during inference.
- **FP8 storage:** tensors occupy eight bits but may be converted to FP16/FP32
  for computation.
- **Native FP8 compute:** kernels consume FP8 operands directly and provide a
  real compute or memory-bandwidth advantage.

Do not report an emulated storage-only path as native FP8. Do not substitute
integer INT8 activation quantization and label it FP8.

## Test environment

Record the following in the result JSON:

- Mac model, chip, GPU-core count, and physical RAM
- macOS version
- Xcode and Swift versions
- Granite-MLX Git commit and dirty-worktree status
- MLX Swift version and revision
- model checkpoint path, source revision, weight format, and group size
- power source and thermal state when available
- exact input path, SHA-256, sample rate, channel count, and duration

The initial reference machine is the 64 GB M1 Max already used for the
quantization benchmark. Results from other Apple Silicon generations must be
stored as separate benchmark environments rather than merged into one timing
series.

## Phase 1: Audit the existing dtype path

- [ ] Add an opt-in diagnostic mode that records shape, dtype, and evaluated
  byte count at these boundaries:
  - frontend output
  - input projection output
  - attention query, key, value, score, probability, and output tensors
  - convolution module input and output
  - feed-forward input and output
  - residual stream after every encoder block
  - middle CTC logits and projected output
  - final logits
- [ ] Confirm whether the frontend currently emits FP32.
- [ ] Confirm the output dtype of quantized matrix multiplication for Q8, Q6,
  and Q5 weights.
- [ ] Identify every deliberate FP32 stability island, including normalization,
  attention softmax, and any reductions.
- [ ] Verify that `MLX.eval` boundaries do not retain tensors from previous
  encoder blocks.
- [ ] Save a machine-readable dtype audit for the 20-second fixture.

The audit must inspect evaluated tensors, not infer the runtime dtype solely
from checkpoint metadata.

## Phase 2: Define activation modes

Implement one explicit runtime option, tentatively:

```text
--activation-precision baseline|fp16|fp8
```

The modes should have these meanings:

### Baseline

Preserve current behavior exactly. This is the control and must reproduce the
existing transcript before testing reduced precision.

### FP16 mixed precision

- Keep audio preprocessing and log-mel feature construction in FP32 initially.
- Cast the frontend output to FP16 immediately before the encoder input
  projection.
- Keep the encoder residual stream, Q/K/V tensors, convolution activations, and
  feed-forward activations in FP16.
- Compute numerically sensitive normalization statistics, reductions, and
  attention softmax in FP32, then cast their outputs back to FP16.
- Test final CTC projection/logits in both FP16 and FP32 during development;
  retain the least expensive mode that preserves output quality.
- Avoid repeated FP16-to-FP32 round trips inside hot paths.

### FP8 experimental

- [ ] Establish which FP8 formats, if any, are exposed by the pinned MLX Swift
  API (for example E4M3- or E5M2-family formats).
- [ ] Establish whether Metal kernels on the M1 Max consume that dtype directly
  or internally promote it.
- [ ] Record supported operators and fallback behavior for quantized matrix
  multiplication, convolution, normalization, residual addition, and softmax.
- [ ] Select the FP8 format from measured range requirements; do not silently
  clamp or overflow activations.
- [ ] Define per-tensor or per-channel scaling and record scale-update policy.
- [ ] Keep normalization, softmax, reductions, and residual accumulation in at
  least FP16, with FP32 statistics where required.
- [ ] Add saturation, underflow, NaN, and infinity counters to diagnostic runs.

If the installed MLX/Metal stack does not provide a genuine and usable FP8
path, stop the FP8 implementation after the capability report. Document the
blocker and estimate the custom-kernel work separately. An emulated FP8 path may
be measured as an experiment, but its result must be labelled `fp8-emulated`.

## Phase 3: Correctness gates

Use the existing deterministic 20-second fixture before any full-lecture run.

- [ ] Confirm baseline output is byte-identical to the existing result.
- [ ] Compare intermediate tensors from baseline and FP16 at every audited
  boundary using maximum absolute error, mean absolute error, RMSE, cosine
  similarity, and relative error percentiles.
- [ ] Compare greedy token IDs, collapsed CTC token IDs, and final transcript.
- [ ] Reject any mode that produces NaN/Inf values or an empty transcript.
- [ ] Inspect the first divergent CTC region rather than relying only on final
  text similarity.
- [ ] Run silence, quiet speech, loud/clipped speech, and a non-16 kHz input
  smoke test.

Proceed to the full recording only after these checks pass.

## Phase 4: Benchmark matrix

Primary weight formats:

- Q8 (expected default)
- Q6
- Q5

Activation modes:

- baseline
- FP16 mixed precision
- FP8 native, if supported
- FP8 emulated, only if useful for diagnosis and clearly labelled

This gives a primary matrix of three weight formats by two required activation
modes, plus supported FP8 variants. Also run converted FP16 weights with
baseline and FP16 activations as a control so weight-quantization kernel effects
can be separated from activation effects.

Inputs:

1. The fixed 20-second WAV fixture for correctness and low-cost iteration.
2. The fixed 6,118.72-second mono 16 kHz WAV for one-pass speed and memory.
3. A fixed chunked version of the same recording, using one documented chunk
   duration and overlap, to measure production-oriented bounded memory.

For each configuration:

- Run one unmeasured warm-up.
- Run at least five short-input measurements in fresh processes.
- Run at least three full-input measurements in fresh processes when practical.
- Randomize configuration order to reduce thermal/order bias.
- Use the same release binary and do not rebuild between comparable runs.
- Synchronize MLX work before stopping inference timers.
- Preserve every transcript and raw process measurement.
- Report medians as the headline result and retain all individual runs.

## Metrics

### Speed

- model load time
- frontend time
- encoder inference time
- decoding time
- process wall time
- real-time factor and audio-seconds-per-second
- median, minimum, maximum, and median absolute deviation across runs

### Memory

- maximum resident set size from `/usr/bin/time -lp`
- macOS peak memory footprint from `/usr/bin/time -lp`
- MLX active memory, peak memory, and cache memory when exposed by the API
- per-stage activation byte estimates from the dtype audit
- one-pass and chunked measurements kept separate

The 44--45 GB footprint and the 1--3 GB maximum RSS are different macOS
measurements; preserve and chart both rather than treating either as ordinary
model size.

### Accuracy relative to baseline

The primary comparison for each reduced activation mode is the same weight
checkpoint with baseline activations. This isolates activation precision from
weight precision.

- raw and normalized transcript equality
- word edit distance and disagreement percentage
- character edit distance and similarity
- BLEU and chrF2 diagnostic scores
- greedy-token and collapsed-CTC-token disagreement
- timestamp/cue differences once Swift timestamp export is available

Also include a secondary comparison against the Swift FP32-weight/baseline-
activation transcript for continuity with the existing quantization benchmark.
None of these similarity measurements is ground-truth WER.

### Numeric health

- NaN and infinity counts
- FP8 saturation and underflow counts
- activation min/max and selected percentiles by stage
- intermediate tensor error metrics versus baseline

## Result files

Store outputs under:

```text
Benchmarks/activation-precision/
  README.md
  results.json
  dtype-audit.json
  transcripts/
  charts/
```

`results.json` is the source of truth and should include a schema version,
environment metadata, configuration definitions, every raw run, summary
statistics, transcript comparisons, failures, and capability notes.

Suggested configuration identity:

```json
{
  "weight_format": "q8",
  "weight_group_size": 64,
  "activation_mode": "fp16-mixed",
  "frontend_dtype": "float32",
  "encoder_dtype": "float16",
  "normalization_statistics_dtype": "float32",
  "softmax_dtype": "float32",
  "logits_dtype": "float16",
  "fp8_format": null,
  "fp8_compute": false
}
```

Generate PNG images directly from `results.json`:

1. full-input inference time by weight and activation precision
2. maximum RSS by weight and activation precision
3. macOS peak memory footprint by weight and activation precision
4. one-pass versus chunked peak memory
5. transcript disagreement versus the same-weight baseline
6. speed-versus-memory trade-off scatter plot
7. intermediate activation error by encoder stage

Charts must include units, hardware, input duration, run count, and whether FP8
is native or emulated. Do not use HTML as the requested chart deliverable.

## Decision criteria

FP16 becomes the default activation mode if it:

- passes all correctness and numeric-health gates,
- materially lowers one-pass peak memory or maximum RSS,
- does not regress median full-input inference time by more than 5%, and
- keeps full-transcript word disagreement against the same-weight baseline
  below 0.1%, unless qualitative review justifies a different threshold.

FP8 remains experimental unless it:

- uses a verified native compute/storage path,
- improves memory or speed beyond FP16,
- is stable across all fixtures, and
- keeps full-transcript word disagreement against the same-weight baseline
  below 0.5% with no catastrophic local regions.

Select the final default using the Q8-weight results first. Q6 and Q5 determine
whether the same activation policy is safe across the other recommended weight
formats.

## Implementation order

- [x] Complete and save the Q8 dtype audit.
- [x] Add the explicit baseline activation mode without changing output.
- [x] Implement FP16 mixed-precision activations.
- [x] Pass the Q8 short-fixture transcript gate.
- [ ] Benchmark Q6/Q5 with baseline and FP16 activations. Q8 is complete and
  determines the default.
- [ ] Benchmark both one-pass and chunked full-input modes.
- [x] Produce the FP8/INT8 capability report through executable QMM probes.
- [x] Implement clearly labelled FP8-E4M3 and signed-INT8 emulated storage
  experiments after finding no native low-precision activation compute path.
- [x] Refine per-tensor scaling and repeat short/full transcript checks.
- [x] Save Q8 JSON, transcripts, environment metadata, and PNG charts.
- [x] Summarize the Q8 recommended default and remaining trade-offs in the
  benchmark README.
