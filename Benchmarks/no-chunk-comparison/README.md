# Swift MLX versus Python MPS without chunking

This benchmark compares the complete 6,118.72-second Stanford CME295 lecture
in one encoder pass. Both runtimes use the source checkpoint's BF16 values cast
to FP16 and disable formatting. Swift uses the converted MLX FP16 checkpoint;
Python uses PyTorch MPS with the safe deferred input cast.

Three paired runs were alternated under the same sustained-load session:

| Runtime | Speech median | Process wall | Peak physical footprint |
|---|---:|---:|---:|
| **Swift MLX FP16, no chunking** | **22.07 s** | **22.59 s** | **13.25 GB** |
| Python MPS FP16, no chunking | 24.87 s | 29.98 s | 14.34 GB |

Swift was 11.3% faster for the paired speech path. Its transcript contained
13,614 words versus Python's 13,595, with 97.6021% word agreement and 98.6674%
character similarity. This difference includes MLX versus PyTorch numerical
behavior and their independent CTC implementations; it is not WER.

The direct answer is therefore yes: removing chunking closes the apparent
Python advantage. With like-for-like FP16 weights in the paired session, Swift
MLX matched and slightly exceeded Python's speed.

Absolute timings varied materially after repeated 14–35 GB one-pass workloads.
Earlier cool-session Python FP16 had a 17.36-second median, while older Swift Q8
one-pass runs ranged from 16.8 to 20.3 seconds. The paired ratio above is more
reliable than comparing absolute medians from different sessions.

This is not a suitable production default. The no-chunk Swift FP16 path used a
13.25 GB process footprint, compared with roughly 1.6–2.1 GB for the bounded
MLX/Core ML production paths. `results.json` preserves all paired run values,
memory counters, transcript hashes, and the output comparison.
