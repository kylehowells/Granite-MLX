#!/usr/bin/env python3
"""Benchmark Parakeet MLX inference with model loading outside the timed region."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path

import mlx.core as mx
from parakeet_mlx.utils import from_pretrained


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v2")
    parser.add_argument("--warmup-audio", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--chunk-duration", type=float, default=120.0)
    parser.add_argument("--overlap-duration", type=float, default=15.0)
    parser.add_argument("--audio-duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    args = parser.parse_args()

    load_start = time.perf_counter()
    model = from_pretrained(args.model, dtype=mx.bfloat16)
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - load_start

    if args.warmup_audio:
        model.transcribe(
            args.warmup_audio,
            dtype=mx.bfloat16,
            chunk_duration=args.chunk_duration or None,
            overlap_duration=args.overlap_duration,
        )

    runs = []
    transcripts = []
    for index in range(args.runs):
        mx.clear_cache()
        started = time.perf_counter()
        result = model.transcribe(
            args.audio,
            dtype=mx.bfloat16,
            chunk_duration=args.chunk_duration or None,
            overlap_duration=args.overlap_duration,
        )
        elapsed = time.perf_counter() - started
        transcript = result.text.strip()
        transcripts.append(transcript)
        run = {
            "index": index + 1,
            "inference_seconds": elapsed,
            "realtime_multiple": args.audio_duration / elapsed,
            "word_count": len(transcript.split()),
            "character_count": len(transcript),
            "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
        }
        runs.append(run)
        print(json.dumps(run), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.transcript.parent.mkdir(parents=True, exist_ok=True)
    args.transcript.write_text(transcripts[0] + "\n")
    times = [run["inference_seconds"] for run in runs]
    report = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "parakeet_mlx": importlib.metadata.version("parakeet-mlx"),
            "mlx": importlib.metadata.version("mlx"),
        },
        "configuration": {
            "model": args.model,
            "dtype": "bfloat16",
            "chunk_duration_seconds": args.chunk_duration or None,
            "overlap_duration_seconds": args.overlap_duration,
        },
        "audio": {
            "path": str(args.audio.resolve()),
            "sha256": file_sha256(args.audio),
            "duration_seconds": args.audio_duration,
        },
        "model_load_seconds": load_seconds,
        "runs": runs,
        "summary": {
            "median_inference_seconds": statistics.median(times),
            "minimum_inference_seconds": min(times),
            "maximum_inference_seconds": max(times),
            "median_realtime_multiple": args.audio_duration / statistics.median(times),
            "transcripts_identical": len(set(transcripts)) == 1,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
