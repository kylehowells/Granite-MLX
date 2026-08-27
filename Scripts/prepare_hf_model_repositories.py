#!/usr/bin/env python3
"""Generate publication model cards and sanitize Granite-MLX manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = ("fp16", "q8", "q6", "q5", "q4")
FAMILIES: dict[str, dict[str, Any]] = {
    "apache": {
        "label": "Apache 2.0",
        "license": "apache-2.0",
        "source_id": "ibm-granite/granite-speech-5.0-470m-turboctc",
        "source_revision": "7e74c6438b7cfb5090cb6a131538f5e8515a7de3",
        "source_sha256": "8b98a8c34fd5fcb081caef719638eded31bb6d197d62053eefc5c1703aaf1ad4",
        "repo_prefix": "granite-speech-5.0-470m-turboctc-mlx-",
        "directory_prefix": "granite-speech-5.0-470m-turboctc-mlx-",
        "directory_overrides": {"q8": "granite-speech-5.0-470m-turboctc-mlx-q8-g128"},
    },
    "non_commercial": {
        "label": "CC BY-NC-SA 4.0 (non-commercial)",
        "license": "cc-by-nc-sa-4.0",
        "source_id": "ibm-granite/granite-speech-5.0-470m-turboctc-nc",
        "source_revision": "0eb7b4fe726a294815dc45d342860465b5af68ef",
        "source_sha256": "e78ded1c2b62a95969abc140f60ae76eb2a869858a1e8174247225304d7fc31b",
        "repo_prefix": "granite-speech-5.0-470m-turboctc-nc-mlx-",
        "directory_prefix": "granite-speech-5.0-470m-turboctc-nc-mlx-",
        "directory_overrides": {},
    },
}


def title(variant: str) -> str:
    return "FP16" if variant == "fp16" else variant.upper()


def table(metadata: dict[str, Any], checkpoints: dict[str, Any], selected: str) -> str:
    source_bytes = checkpoints["source"]["model_file_bytes"]
    rows = [
        "| Variant | Repository | Weight file | Size vs source | Agreement with source |",
        "|---|---|---:|---:|---:|",
        (
            f"| IBM source | [{metadata['source_id']}](https://huggingface.co/{metadata['source_id']}) "
            f"| {source_bytes / 1048576:.2f} MiB | 100.00% | 100.0000% |"
        ),
    ]
    for variant in VARIANTS:
        checkpoint = checkpoints[variant]
        repo = f"iky1e/{metadata['repo_prefix']}{variant}"
        label = title(variant)
        if variant == selected:
            label = f"**{label} (this repository)**"
        rows.append(
            f"| {label} | [{repo}](https://huggingface.co/{repo}) "
            f"| {checkpoint['model_file_bytes'] / 1048576:.2f} MiB "
            f"| {100 * checkpoint['model_file_bytes'] / source_bytes:.2f}% "
            f"| {checkpoint['accuracy_vs_source']['word_agreement_percent']:.4f}% |"
        )
    return "\n".join(rows)


def quantization_description(variant: str) -> str:
    if variant == "fp16":
        return "FP16 converted weights (no weight quantization)"
    if variant == "q8":
        return "8-bit affine weight quantization, group size 128 with group size 64 for `encoder.input_linear`"
    return f"{variant[1:]}-bit affine weight quantization, group size 64"


def conversion_command(metadata: dict[str, Any], variant: str) -> str:
    quantization = ""
    if variant != "fp16":
        group_size = 128 if variant == "q8" else 64
        quantization = (
            f" \\\n  --quantization-bits {variant[1:]} \\\n  --group-size {group_size}"
        )
    return f"""uv run python Scripts/convert_granite.py \\
  /path/to/source-checkpoint \\
  /path/to/output-{variant} \\
  --precision fp16{quantization} \\
  --source-model-id {metadata['source_id']} \\
  --source-revision {metadata['source_revision']}"""


def card(metadata: dict[str, Any], checkpoints: dict[str, Any], variant: str) -> str:
    source = metadata["source_id"]
    repo = f"iky1e/{metadata['repo_prefix']}{variant}"
    nc_warning = ""
    if metadata["license"] == "cc-by-nc-sa-4.0":
        nc_warning = (
            "> [!IMPORTANT]\n"
            "> These converted weights are for non-commercial use under CC BY-NC-SA 4.0. "
            "For commercial-compatible weights, use the Apache 2.0 family instead.\n\n"
        )
    return f"""---
license: {metadata['license']}
language:
- en
pipeline_tag: automatic-speech-recognition
library_name: mlx
base_model: {source}
tags:
- mlx
- swift
- granite
- speech-to-text
- automatic-speech-recognition
---

# Granite Speech 5.0 470M TurboCTC MLX — {title(variant)}

{nc_warning}This repository contains **{quantization_description(variant)}** for native Apple-silicon inference with the Granite-MLX Swift runtime.

It is converted from IBM's [{source}](https://huggingface.co/{source}) at revision [`{metadata['source_revision']}`](https://huggingface.co/{source}/tree/{metadata['source_revision']}). The source `model.safetensors` SHA-256 is `{metadata['source_sha256']}`.

The conversion transposes PyTorch depthwise `Conv1d` kernels into MLX layout, removes training-only batch counters, converts retained floating-point tensors to FP16, and applies weight-only affine quantization where applicable. Activations remain floating point at runtime.

## Family comparison

{table(metadata, checkpoints, variant)}

“Agreement with source” is word-level transcript agreement, calculated as `100 − Levenshtein word edits / source words`. It is **not WER** and does not measure correctness against a human transcript. The reference is the matching original IBM checkpoint loaded by the same native Swift runtime.

The test recording is a 6,118.72-second (101m58.72s) single-speaker Stanford CME295 lecture. All checkpoints used Granite-MLX's bounded-memory defaults: 122.88-second chunks, 20.48-second context, FP16 activations, greedy CTC decoding, and a 64 MiB MLX cache. The source transcript contained {checkpoints['source']['accuracy_vs_source']['reference_words']:,} words. Raw benchmark JSON and transcripts are preserved with the Granite-MLX project and will be published with its source repository.

## Usage

```shell
granite-mlx /path/to/audio-or-video --model {repo}
```

Granite-MLX accepts common audio and video files, downloading this repository automatically on first use. It can export TXT, SRT, WebVTT, JSON, or all formats.

## Reproducing the conversion

```shell
{conversion_command(metadata, variant)}
```

The publication converter (`Scripts/convert_granite.py`) has SHA-256 `83141299b6e680fbdc020aed6674683b5cac03685dec839a30767f55de0942fb`. Validation used Granite-MLX's native Swift release build with Apple Swift 6.2.3, MLX Swift 0.31.4 (`dc43e62d7055353c7f99fa071a4e71d29dfddc44`), swift-transformers 1.3.3 (`2fa33e1f5e7131a7fc64c28e6d161dcec0d24820`), macOS 26.5.2, and Xcode 26.2.

## License

These converted **model weights** retain the source model's {metadata['label']} terms. The Granite-MLX Swift software is separate from the weights; this model repository does not set the software's license.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())

    publication: dict[str, Any] = {"repositories": []}
    for family_name, metadata in FAMILIES.items():
        checkpoints = results["families"][family_name]["checkpoints"]
        for variant in VARIANTS:
            directory_name = metadata["directory_overrides"].get(variant) or metadata["directory_prefix"] + variant
            directory = args.model_root / directory_name
            manifest_path = directory / "granite-mlx-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["source"] = {
                "model_id": metadata["source_id"],
                "revision": metadata["source_revision"],
            }
            manifest["source_weights"] = {
                "path": "model.safetensors",
                "sha256": metadata["source_sha256"],
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            (directory / "README.md").write_text(card(metadata, checkpoints, variant))
            publication["repositories"].append({
                "repo_id": f"iky1e/{metadata['repo_prefix']}{variant}",
                "local_directory": directory.name,
                "family": family_name,
                "variant": variant,
                "license": metadata["license"],
            })

    output = args.results.parent / "repositories.json"
    output.write_text(json.dumps(publication, indent=2) + "\n")
    print(f"Wrote ten model cards and {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
