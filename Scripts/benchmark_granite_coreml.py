#!/usr/bin/env python3
"""Benchmark a fixed-shape Granite Core ML package and verify PyTorch parity."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from safetensors.numpy import load_file
from tokenizers import Tokenizer

from Scripts.granite_coreml_model import load_granite_coreml_model


COMPUTE_UNITS = {
    "all": ct.ComputeUnit.ALL,
    "cpu-gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu-ne": ct.ComputeUnit.CPU_AND_NE,
    "cpu": ct.ComputeUnit.CPU_ONLY,
}


def ctc_tokens(frame_ids: np.ndarray, blank_id: int = 0) -> list[int]:
    """Collapse repeated CTC frames and remove blank tokens."""
    tokens: list[int] = []
    previous: int | None = None
    for value in frame_ids.reshape(-1):
        token = int(value)
        if token != previous and token != blank_id:
            tokens.append(token)
        previous = token
    return tokens


def decode(tokenizer: Tokenizer, frame_ids: np.ndarray) -> str:
    """Decode greedy CTC frame IDs with the source Granite tokenizer."""
    return tokenizer.decode(ctc_tokens(frame_ids), skip_special_tokens=True).strip()


def padded_features(path: Path, frame_count: int) -> np.ndarray:
    """Load dumped frontend features and pad them to the fixed model shape."""
    features = load_file(path)["features"].astype(np.float32)
    if features.shape[1] > frame_count:
        raise ValueError(
            f"Input has {features.shape[1]} frames but model accepts {frame_count}")
    return np.pad(features, ((0, 0), (0, frame_count - features.shape[1]), (0, 0)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("source_model", type=Path)
    parser.add_argument("--compute-units", choices=COMPUTE_UNITS, default="all")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metadata_path = args.package.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    frame_count = int(metadata["feature_frames"])
    features = padded_features(args.features, frame_count)
    tokenizer = Tokenizer.from_file(str(args.source_model / "tokenizer.json"))

    output_mode = metadata.get("output_mode", "argmax")
    output_name = {"argmax": "frame_ids", "logits": "logits", "hidden": "hidden"}[output_mode]
    expected: np.ndarray | None = None
    pytorch_seconds: float | None = None
    if not args.skip_pytorch:
        reference = load_granite_coreml_model(args.source_model)
        started = time.perf_counter()
        with torch.inference_mode():
            tensor = torch.from_numpy(features)
            if output_mode == "argmax":
                expected = reference(tensor).numpy()
            elif output_mode == "logits":
                expected = reference.encoder(tensor).numpy()
            else:
                expected = reference.encoder.encode(tensor).numpy()
        pytorch_seconds = time.perf_counter() - started
        del reference

    load_started = time.perf_counter()
    model = ct.models.MLModel(
        str(args.package), compute_units=COMPUTE_UNITS[args.compute_units])
    load_seconds = time.perf_counter() - load_started

    durations: list[float] = []
    actual: np.ndarray | None = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        prediction = model.predict({"features": features.astype(np.float16)})
        durations.append(time.perf_counter() - started)
        actual = np.asarray(prediction[output_name])
    assert actual is not None

    result = {
        "package": str(args.package),
        "compute_units": args.compute_units,
        "feature_frames": frame_count,
        "audio_seconds": frame_count / 50,
        "load_seconds": load_seconds,
        "prediction_seconds": durations,
        "warm_median_seconds": float(np.median(durations[1:])),
        "warm_realtime_factor": float(np.median(durations[1:]) / (frame_count / 50)),
        "warm_audio_speed_x": float((frame_count / 50) / np.median(durations[1:])),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "text": decode(tokenizer, actual) if output_mode == "argmax" else None,
        "pytorch_seconds": pytorch_seconds,
    }
    if expected is not None:
        if output_mode == "argmax":
            result.update({
                "frame_disagreement_fraction": float(np.mean(actual != expected)),
                "pytorch_text": decode(tokenizer, expected),
                "text_matches_pytorch": decode(tokenizer, actual) == decode(tokenizer, expected),
            })
        else:
            difference = actual.astype(np.float32) - expected.astype(np.float32)
            result.update({
                "maximum_absolute_error": float(np.max(np.abs(difference))),
                "mean_absolute_error": float(np.mean(np.abs(difference))),
            })
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
