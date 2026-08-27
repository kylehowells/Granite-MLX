#!/usr/bin/env python3
"""Prepare MLX punctuation model cards and a publication manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


MODEL_ROOT = Path("/Users/kylehowells/Developer/ML-Models")
PROJECT = Path(__file__).resolve().parents[1]
VARIANTS = ("fp16", "q8", "q6", "q5", "q4")
PREFIX = "punctuation-fullstop-truecase-english-mlx-"
SOURCE_ID = "1-800-BAD-CODE/punctuation_fullstop_truecase_english"
SOURCE_REVISION = "b26fd1c40e88678859048898218ea4edcc24c84a"
SOURCE_BYTES = 209_532_928


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def label(variant: str) -> str:
    return "FP16" if variant == "fp16" else variant.upper()


def metrics() -> dict[str, dict]:
    values = {}
    for variant in VARIANTS:
        result = json.loads(
            (PROJECT / "Benchmarks" / "punctuation" / f"mlx-{variant}.json").read_text()
        )
        directory = MODEL_ROOT / f"punctuation_fullstop_truecase_english-mlx-{variant}"
        weight = directory / "model.safetensors"
        values[variant] = {
            "weight_bytes": weight.stat().st_size,
            "weight_sha256": sha256(weight),
            "inference_seconds": result["inference_seconds"],
            "max_rss_bytes": result["max_rss_bytes"],
            "character_agreement_percent": 100 - result["character_disagreement_percent"],
            "word_agreement_percent": 100 - result["word_disagreement_percent"],
            "sentence_count": result["sentence_count"],
        }
    return values


def comparison_table(values: dict[str, dict], selected: str) -> str:
    rows = [
        "| Variant | Repository | Weight file | Size vs ONNX FP32 | Character agreement | Word agreement | Formatter inference |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| ONNX FP32 source | [{SOURCE_ID}](https://huggingface.co/{SOURCE_ID}) | {SOURCE_BYTES / 1_000_000:.1f} MB | 100.0% | 100.0000% | 100.0000% | 2.044s |",
    ]
    for variant in VARIANTS:
        value = values[variant]
        title = label(variant)
        if variant == selected:
            title = f"**{title} (this repository)**"
        repo = f"iky1e/{PREFIX}{variant}"
        rows.append(
            f"| {title} | [{repo}](https://huggingface.co/{repo}) "
            f"| {value['weight_bytes'] / 1_000_000:.1f} MB "
            f"| {100 * value['weight_bytes'] / SOURCE_BYTES:.1f}% "
            f"| {value['character_agreement_percent']:.4f}% "
            f"| {value['word_agreement_percent']:.4f}% "
            f"| {value['inference_seconds']:.3f}s |"
        )
    return "\n".join(rows)


def model_card(values: dict[str, dict], variant: str) -> str:
    repo = f"iky1e/{PREFIX}{variant}"
    precision = (
        "FP16 converted weights without weight quantization"
        if variant == "fp16"
        else f"{variant[1:]}-bit affine weight quantization with group size 64 and FP16 residual tensors"
    )
    bits_argument = "" if variant == "fp16" else " \\" + "\n  --bits " + variant[1:]
    return f"""---
license: apache-2.0
language:
- en
library_name: mlx
base_model: {SOURCE_ID}
pipeline_tag: text-classification
tags:
- mlx
- punctuation-restoration
- capitalization
- sentence-boundary-detection
- swift
---

# English punctuation, capitalization, and segmentation MLX — {label(variant)}

This repository contains **{precision}** for Apple-silicon inference with MLX.

It is converted from [{SOURCE_ID}](https://huggingface.co/{SOURCE_ID}) at revision [`{SOURCE_REVISION}`](https://huggingface.co/{SOURCE_ID}/tree/{SOURCE_REVISION}). The original model restores punctuation and capitalization and predicts sentence boundaries for lowercase English text. Its source ONNX SHA-256 is `dd922d459da618cd324280889740608b76fb3e9e61d3f402291be1251f91421b`.

## MLX variant comparison

{comparison_table(values, variant)}

Agreement is measured against the original ONNX FP32 model's formatted output, not against a human transcript. Character and word agreement are `100 − normalized Levenshtein distance`. The input was Granite Q8's raw transcript of a 6,118.72-second (101m58.72s) Stanford CME295 lecture: 69,168 input characters and 71,116 ONNX-formatted characters. Formatter time excludes process startup and model loading. Peak RSS from the Python harness includes Python and framework overhead.

Q8 is the recommended default: it is substantially smaller than FP16 while remaining very close to ONNX FP32 and was the fastest measured MLX variant on this machine.

## Granite-MLX usage

Formatted output is intended to be the Granite-MLX default:

```shell
granite-mlx recording.mp4 --punctuation-model {repo}
```

Use `--no-punctuate` when exact raw Granite CTC text or minimum memory usage is required.

## Files

- `model.safetensors`: MLX weights
- `mlx_config.json`: architecture, precision, quantization, and source metadata
- `tokenizer.json` and `tokenizer_config.json`: native-compatible SentencePiece Unigram tokenizer
- `spe_32k_lc_en.model`: original SentencePiece model for parity/reference runtimes
- `config.yaml`: source labels and sequence configuration

## Reproducing conversion

```shell
uv run python Scripts/convert_punctuation.py \\
  /path/to/punctuation_fullstop_truecase_english \\
  /path/to/output-{variant}{bits_argument}
```

The converter and complete benchmark artifacts are maintained in the [Granite-MLX project](https://github.com/kylehowells/Granite-MLX).

## License

These converted model weights retain the original model's Apache 2.0 license. Granite-MLX is separate software that downloads and runs a user-selected checkpoint.
"""


def main() -> None:
    values = metrics()
    publication = {"source_model": SOURCE_ID, "repositories": []}
    for variant in VARIANTS:
        directory = MODEL_ROOT / f"punctuation_fullstop_truecase_english-mlx-{variant}"
        (directory / "README.md").write_text(model_card(values, variant), encoding="utf-8")
        publication["repositories"].append(
            {
                "variant": variant,
                "repo_id": f"iky1e/{PREFIX}{variant}",
                "local_directory": str(directory),
                **values[variant],
            }
        )
    output = PROJECT / "Benchmarks" / "punctuation" / "repositories.json"
    output.write_text(json.dumps(publication, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
