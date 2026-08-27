#!/usr/bin/env python3
"""Convert Granite Speech 5.0 CTC weights into the MLX runtime layout.

The checkpoint is already safetensors, so conversion is primarily a manifest
and layout-normalization step. PyTorch Conv1d stores weights as
[out_channels, in_channels, kernel], while MLX Conv1d consumes
[out_channels, kernel, in_channels]. Linear and normalization tensors retain
their shapes. The source model ID/path is intentionally configurable so both
Granite 5.0 weight-license variants can be converted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from safetensors.torch import load_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_quantizable_weight(key: str, ndim: int) -> bool:
    """Return whether MLXNN will replace this parameter's module.

    Granite's only two-dimensional weights belong to Linear or Embedding
    modules. Convolution and normalization weights are one- or three-
    dimensional and intentionally remain floating point.
    """

    return key.endswith(".weight") and ndim == 2


def convert(
    source: Path,
    destination: Path,
    precision: str,
    quantization_bits: int | None = None,
    group_size: int = 64,
    source_model_id: str | None = None,
    source_revision: str | None = None,
) -> None:
    source_weights = source / "model.safetensors"
    if not source_weights.is_file():
        raise FileNotFoundError(f"Missing model.safetensors in {source}")
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {source}")

    destination.mkdir(parents=True, exist_ok=True)
    if quantization_bits is not None and precision != "fp16":
        raise ValueError("Quantized checkpoints currently require --precision fp16")

    source_tensors = load_file(str(source_weights), device="cpu")
    converted: dict[str, object] = {}
    skipped: list[str] = []
    transformed: list[str] = []
    quantized: list[str] = []
    group_sizes: dict[str, int] = {}

    for key, value in source_tensors.items():
        # This is optimizer/training bookkeeping, not an inference parameter.
        if key.endswith(".num_batches_tracked"):
            skipped.append(key)
            continue
        # Convert BF16 source tensors through FP32 because NumPy/safetensors
        # cannot represent BF16 directly. MLX can cast the resulting artifact
        # to BF16 at runtime.
        array = value.detach().float().cpu().numpy()
        if key.endswith(".conv.depthwise_conv.weight") and array.ndim == 3:
            array = np.transpose(array, (0, 2, 1))
            transformed.append(key)
        if precision == "fp16" and array.dtype == np.float32:
            array = array.astype(np.float16)
        elif precision == "bf16":
            # safetensors/numpy has no native bfloat16. Keep FP32 and let MLX
            # cast at load time; this preserves exact conversion artifacts.
            pass
        converted[key] = array

    if quantization_bits is not None:
        import mlx.core as mx

        quantized_arrays: dict[str, mx.array] = {}
        for key, value in converted.items():
            array = mx.array(value)
            if is_quantizable_weight(key, array.ndim):
                # Granite's 320-wide input projection cannot use group 128.
                # Select the largest supported group no larger than the
                # requested one which divides this particular matrix.
                tensor_group_size = next(
                    candidate
                    for candidate in (group_size, 64, 32)
                    if candidate <= group_size and array.shape[-1] % candidate == 0
                )
                packed, scales, biases = mx.quantize(
                    array,
                    group_size=tensor_group_size,
                    bits=quantization_bits,
                    mode="affine",
                )
                base = key.removesuffix(".weight")
                mx.eval(packed, scales, biases)
                quantized_arrays[key] = packed
                quantized_arrays[f"{base}.scales"] = scales
                quantized_arrays[f"{base}.biases"] = biases
                quantized.append(key)
                if tensor_group_size != group_size:
                    group_sizes[base] = tensor_group_size
            else:
                mx.eval(array)
                quantized_arrays[key] = array
        converted = quantized_arrays

    output_weights = destination / "model.safetensors"
    if quantization_bits is None:
        save_file(converted, str(output_weights), metadata={"format": "mlx"})
    else:
        import mlx.core as mx

        mx.save_safetensors(
            str(output_weights), converted, metadata={"format": "mlx"}
        )

    copied = []
    for name in (
        "config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
        "processor_config.json", "generation_config.json", "README.md",
    ):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
            copied.append(name)

    # swift-transformers implements this tokenizer.json as ordinary byte-level
    # BPE, but does not register IBM's Python-only ParakeetTokenizer class name.
    # Record the equivalent built-in class in the converted artifact so native
    # loading remains strict and does not write warnings to transcript stdout.
    tokenizer_config_path = destination / "tokenizer_config.json"
    if tokenizer_config_path.is_file():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        tokenizer_config["tokenizer_class"] = "GPT2Tokenizer"
        tokenizer_config_path.write_text(
            json.dumps(tokenizer_config, indent=2) + "\n", encoding="utf-8"
        )

    if quantization_bits is not None:
        converted_config_path = destination / "config.json"
        converted_config = json.loads(converted_config_path.read_text(encoding="utf-8"))
        converted_config["quantization"] = {
            "group_size": group_size,
            "group_sizes": group_sizes,
            "bits": quantization_bits,
            "mode": "affine",
        }
        converted_config_path.write_text(
            json.dumps(converted_config, indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        "format": "granite-mlx",
        "source": {
            "model_id": source_model_id,
            "revision": source_revision,
        },
        "source_weights": {"path": "model.safetensors", "sha256": sha256(source_weights)},
        "precision": precision,
        "quantization": None if quantization_bits is None else {
            "group_size": group_size,
            "group_sizes": group_sizes,
            "bits": quantization_bits,
            "mode": "affine",
            "quantized_tensor_count": len(quantized),
        },
        "weight_file": "model.safetensors",
        "skipped_tensors": skipped,
        "transposed_tensors": transformed,
        "copied_metadata": copied,
        "notes": [
            "Linear and normalization tensors preserve source shapes.",
            "PyTorch Conv1d depthwise weights are transposed [O,I,K] -> [O,K,I] for MLX.",
            "BF16 conversion is deferred to MLX load because NumPy safetensors has no native BF16 dtype.",
            "Quantization, when requested, is affine weight-only quantization; activations remain floating point.",
        ],
    }
    (destination / "granite-mlx-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_weights} ({len(converted)} tensors)")
    print(
        f"Skipped {len(skipped)} training-only tensors; transformed {len(transformed)} tensors; "
        f"quantized {len(quantized)} matrix weights"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source Granite model directory")
    parser.add_argument("destination", type=Path, help="Output MLX model directory")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument(
        "--quantization-bits", type=int, choices=[2, 3, 4, 5, 6, 8],
        help="Apply affine weight-only quantization at this bit width",
    )
    parser.add_argument(
        "--group-size", type=int, choices=[32, 64, 128], default=64,
        help="Number of matrix values sharing each quantization scale/bias",
    )
    parser.add_argument(
        "--source-model-id",
        help="Public source repository recorded in publication metadata",
    )
    parser.add_argument(
        "--source-revision",
        help="Exact source repository revision recorded in publication metadata",
    )
    args = parser.parse_args()
    convert(
        args.source.expanduser().resolve(),
        args.destination.expanduser().resolve(),
        args.precision,
        quantization_bits=args.quantization_bits,
        group_size=args.group_size,
        source_model_id=args.source_model_id,
        source_revision=args.source_revision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
