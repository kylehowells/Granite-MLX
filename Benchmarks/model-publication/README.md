# Hugging Face model publication benchmark

This benchmark compares each converted MLX checkpoint with its own IBM source
checkpoint on the same 6,118.72-second (101m58.72s) Stanford CME295 lecture.
It uses the native Swift release runtime with the CLI defaults: FP16
activations, 122.88-second chunks, 20.48-second context, greedy CTC decoding,
and a 64 MiB MLX cache.

Agreement is `100 - word Levenshtein edits / source words`. It measures
conversion/quantization similarity, not accuracy against a human transcript
and not ground-truth WER. Each checkpoint was run once on the full recording.

## Apache 2.0 weight family

| Variant | Weight file | Size vs source | Word agreement | Word edits / source words |
|---|---:|---:|---:|---:|
| IBM BF16 source | 902.35 MiB | 100.00% | 100.0000% | 0 / 13,615 |
| [FP16](https://huggingface.co/iky1e/granite-speech-5.0-470m-turboctc-mlx-fp16) | 902.22 MiB | 99.99% | 99.9706% | 4 / 13,615 |
| [Q8 mixed G128/G64](https://huggingface.co/iky1e/granite-speech-5.0-470m-turboctc-mlx-q8) | 466.03 MiB | 51.65% | 99.8825% | 16 / 13,615 |
| [Q6 G64](https://huggingface.co/iky1e/granite-speech-5.0-470m-turboctc-mlx-q6) | 367.51 MiB | 40.73% | 99.6989% | 41 / 13,615 |
| [Q5 G64](https://huggingface.co/iky1e/granite-speech-5.0-470m-turboctc-mlx-q5) | 311.22 MiB | 34.49% | 99.4712% | 72 / 13,615 |
| [Q4 G64](https://huggingface.co/iky1e/granite-speech-5.0-470m-turboctc-mlx-q4) | 254.93 MiB | 28.25% | 98.4135% | 216 / 13,615 |

## Non-commercial weight family

| Variant | Weight file | Size vs source | Word agreement | Word edits / source words |
|---|---:|---:|---:|---:|
| IBM BF16 source | 902.35 MiB | 100.00% | 100.0000% | 0 / 13,618 |
| [FP16](https://huggingface.co/iky1e/granite-speech-5.0-470m-turboctc-nc-mlx-fp16) | 902.22 MiB | 99.99% | 99.9853% | 2 / 13,618 |
| [Q8 mixed G128/G64](https://huggingface.co/iky1e/granite-speech-5.0-470m-turboctc-nc-mlx-q8) | 466.03 MiB | 51.65% | 99.8972% | 14 / 13,618 |
| [Q6 G64](https://huggingface.co/iky1e/granite-speech-5.0-470m-turboctc-nc-mlx-q6) | 367.51 MiB | 40.73% | 99.6328% | 50 / 13,618 |
| [Q5 G64](https://huggingface.co/iky1e/granite-speech-5.0-470m-turboctc-nc-mlx-q5) | 311.22 MiB | 34.49% | 99.4639% | 73 / 13,618 |
| [Q4 G64](https://huggingface.co/iky1e/granite-speech-5.0-470m-turboctc-nc-mlx-q4) | 254.93 MiB | 28.25% | 98.8545% | 156 / 13,618 |

[`results.json`](results.json) contains timings, memory measurements, file
hashes, BLEU, chrF2, character similarity, word edits, and agreement values.
[`repositories.json`](repositories.json) records the ten repository IDs and
the checkpoint directory names used by the publication tooling. The
accompanying text files are the decoded transcripts used for the comparisons.

The Swift CLI's published Q8 default was also tested from an isolated empty
Hugging Face cache. It downloaded the repository without a `--model` argument,
loaded the checkpoint, and transcribed the fixed 20-second fixture at 171.6x
realtime inference on this machine. The initial 488 MB download took 921.2
seconds on the test connection; download time is not model inference time.
