#!/usr/bin/env python3
"""Apply Core ML weight compression to a converted Granite ML Program."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import coremltools as ct


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--method", choices=["linear", "palette-uniform", "palette-kmeans"],
        default="linear")
    parser.add_argument("--bits", type=int, choices=[4, 6, 8], default=8)
    parser.add_argument(
        "--granularity", choices=["per-channel", "per-block", "per-tensor"],
        default="per-block")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--weight-threshold", type=int, default=2048)
    args = parser.parse_args()

    if args.method == "linear" and args.bits not in (4, 8):
        parser.error("Core ML linear weight quantization supports 4 or 8 bits")

    started = time.perf_counter()
    model = ct.models.MLModel(str(args.source), skip_model_load=True)
    if args.method == "linear":
        granularity = args.granularity.replace("-", "_")
        kwargs: dict[str, object] = {
            "mode": "linear_symmetric",
            "dtype": f"int{args.bits}",
            "granularity": granularity,
            "weight_threshold": args.weight_threshold,
        }
        if granularity == "per_block":
            kwargs["block_size"] = args.block_size
        operation = ct.optimize.coreml.OpLinearQuantizerConfig(**kwargs)
        compressed = ct.optimize.coreml.linear_quantize_weights(
            model,
            ct.optimize.coreml.OptimizationConfig(global_config=operation),
        )
    else:
        if args.granularity == "per-block":
            parser.error("Palette compression uses per-tensor or per-channel granularity")
        operation = ct.optimize.coreml.OpPalettizerConfig(
            mode=args.method.removeprefix("palette-"),
            nbits=args.bits,
            granularity=(
                "per_grouped_channel"
                if args.granularity == "per-channel" else "per_tensor"
            ),
            group_size=args.block_size,
            weight_threshold=args.weight_threshold,
        )
        compressed = ct.optimize.coreml.palettize_weights(
            model,
            ct.optimize.coreml.OptimizationConfig(global_config=operation),
        )

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    compressed.save(str(args.destination))
    source_metadata_path = args.source.with_suffix(".json")
    source_metadata = (
        json.loads(source_metadata_path.read_text())
        if source_metadata_path.is_file() else {}
    )
    result = {
        **source_metadata,
        "source": str(args.source),
        "destination": str(args.destination),
        "method": args.method,
        "bits": args.bits,
        "granularity": args.granularity,
        "block_size": args.block_size,
        "weight_threshold": args.weight_threshold,
        "seconds": time.perf_counter() - started,
    }
    args.destination.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
