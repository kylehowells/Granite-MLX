#!/usr/bin/env python3
"""Convert Granite Speech 5.0 TurboCTC to a fixed-shape Core ML program."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from Scripts.granite_coreml_model import (
    GraniteCoreMLHiddenModel,
    load_granite_coreml_model,
)


def convert(
    source: Path,
    destination: Path,
    feature_frames: int,
    output_mode: str,
    minimum_target: str,
) -> None:
    started = time.perf_counter()
    model = load_granite_coreml_model(source)
    conversion_model: torch.nn.Module
    if output_mode == "logits":
        conversion_model = model.encoder
    elif output_mode == "hidden":
        conversion_model = GraniteCoreMLHiddenModel(model.encoder)
    else:
        conversion_model = model
    example = torch.zeros((1, feature_frames, 320), dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(
            conversion_model, example, strict=False, check_trace=False)
    traced_seconds = time.perf_counter() - started

    destination.parent.mkdir(parents=True, exist_ok=True)
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target={
            "macos14": ct.target.macOS14,
            "macos15": ct.target.macOS15,
        }[minimum_target],
        compute_precision=ct.precision.FLOAT16,
        inputs=[ct.TensorType(
            name="features", shape=example.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name={
            "argmax": "frame_ids",
            "logits": "logits",
            "hidden": "hidden",
        }[output_mode])],
    )
    converted.author = "Granite-MLX"
    converted.short_description = "Granite Speech 5.0 TurboCTC fixed-shape encoder"
    converted.user_defined_metadata.update({
        "source_model": str(source),
        "feature_frames": str(feature_frames),
        "audio_seconds": str(feature_frames / 50),
        "output_mode": output_mode,
        "precision": "fp16",
        "minimum_target": minimum_target,
    })
    converted.save(destination)
    metadata = {
        "source": str(source),
        "destination": str(destination),
        "feature_frames": feature_frames,
        "audio_seconds": feature_frames / 50,
        "output_frames": feature_frames // 4,
        "output_mode": output_mode,
        "precision": "fp16",
        "minimum_target": minimum_target,
        "trace_seconds": traced_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--feature-frames", type=int, default=1024,
        help="Fixed frontend-frame count; Granite emits 50 feature frames/s.")
    parser.add_argument(
        "--output-mode", choices=["argmax", "logits", "hidden"], default="argmax")
    parser.add_argument(
        "--minimum-target", choices=["macos14", "macos15"], default="macos14",
        help="Core ML deployment target; blockwise compression requires macOS 15.")
    args = parser.parse_args()
    if args.feature_frames <= 0 or args.feature_frames % 512:
        parser.error("--feature-frames must be a positive multiple of 512")
    convert(
        args.source.expanduser().resolve(),
        args.destination.expanduser().resolve(),
        args.feature_frames,
        args.output_mode,
        args.minimum_target,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
