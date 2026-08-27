#!/usr/bin/env python3
"""Calibrate and apply Core ML INT8 activation quantization to Granite."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from safetensors.numpy import load_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("features", nargs="+", type=Path)
    parser.add_argument("--calibration-op-group-size", type=int, default=64)
    args = parser.parse_args()

    metadata = json.loads(args.source.with_suffix(".json").read_text())
    frame_count = int(metadata["feature_frames"])
    samples: list[dict[str, np.ndarray]] = []
    for path in args.features:
        features = load_file(path)["features"].astype(np.float16)
        if features.shape[1] > frame_count:
            features = features[:, :frame_count]
        else:
            features = np.pad(
                features,
                ((0, 0), (0, frame_count - features.shape[1]), (0, 0)))
        samples.append({"features": features})

    started = time.perf_counter()
    model = ct.models.MLModel(str(args.source), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    config = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric"))
    quantized = ct.optimize.coreml.linear_quantize_activations(
        model,
        config,
        samples,
        calibration_op_group_size=args.calibration_op_group_size,
    )
    quantized.save(str(args.destination))
    result = {
        **metadata,
        "source": str(args.source),
        "destination": str(args.destination),
        "activation_precision": "int8",
        "calibration_samples": [str(path) for path in args.features],
        "calibration_op_group_size": args.calibration_op_group_size,
        "seconds": time.perf_counter() - started,
    }
    args.destination.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
