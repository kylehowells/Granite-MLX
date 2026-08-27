# Bounded MLX profile optimization

This experiment tests whether Granite MLX Q8 can run faster without giving up
the bounded profile's approximately 1.7 GB peak footprint. Each configuration
transcribed the complete 6,118.72-second (101m58.72s) Stanford CME295 lecture
three times in rotated order on an Apple M1 Max. All runs used FP16 encoder
activations, a 64 MiB MLX cache limit, raw CTC output, and the mixed-G128/G64
Q8 checkpoint.

The table reports medians from three fresh processes. Agreement is measured
against the prior 122.88-second/20.48-second profile. It is an implementation
diagnostic rather than WER against a human transcript.

| Profile | Central audio | Context per side | Speech inference | Process wall | Peak footprint | Word agreement |
|---|---:|---:|---:|---:|---:|---:|
| Prior default | 122.88 s | 20.48 s | 25.281 s | 25.73 s | 1.652 GB | 100.0000% |
| **Selected default** | **122.88 s** | **10.24 s** | **21.952 s** | **22.40 s** | **1.627 GB** | **99.9045%** |
| Larger centre, same target memory | 143.36 s | 10.24 s | 21.829 s | 22.26 s | 1.653 GB | 99.7428% |

Reducing context to one 10.24-second attention block was 13.2% faster and used
25.3 MB less peak physical memory in the paired experiment. It changed 13
words across the prior transcript's 13,610 words. The larger-centre profile
saved only another 0.6% inference time while using the old amount of memory and
changing more words, so it was rejected.

## Runtime prototypes

Two MLX runtime changes were also tested with the selected chunk geometry.

| Prototype | Speech inference | Peak footprint | Result |
|---|---:|---:|---|
| Baseline | 22.151 s | 1.627 GB | Selected implementation |
| Retain MLX cache between chunks | 22.685 s | 1.667 GB | Slower and 39.9 MB larger |
| Compile each Conformer layer | 22.659 s | 1.628 GB | Slower |
| Compile layers and retain cache | 21.935 s | 1.682 GB | 1.0% faster but 54.6 MB larger |

Every profile and prototype produced a byte-identical transcript across its
three rounds. The compilation/cache experiments were not retained because no
variant delivered a useful speed improvement without increasing memory.

## Measurement conditions

macOS background media analysis and indexing were active during this session.
Load average rose from 18.23 to 35.03 during the profile matrix and from 25.41
to 64.54 during the prototype matrix. Rotated interleaving makes the relative
comparison useful, but these absolute times should not replace the quieter
three-backend release matrix.

[results.json](results.json) contains every run, timing and memory counter,
transcript hash, comparison, and the selected default. Deterministic raw
transcripts are under [transcripts](transcripts).

## Selected CLI profile

```bash
granite-mlx lecture.wav \
  --model /path/to/granite-speech-5.0-470m-turboctc-mlx-q8 \
  --audio-chunk-duration 122.88 \
  --audio-chunk-context 10.24 \
  --mlx-cache-limit-mb 64 \
  --no-punctuate
```

Those chunk and context values are now the MLX defaults. Core ML retains its
separate graph-dependent chunk sizing and up to 20.48 seconds of context.
