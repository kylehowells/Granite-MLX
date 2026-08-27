#!/usr/bin/env python3
"""Benchmark IBM Granite Speech source weights through the documented Python API."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from granite_reference import load_audio, memory_report


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def synchronize(device: str) -> None:
    """Wait for queued accelerator work before stopping a timing interval."""

    if device == "mps":
        import torch

        torch.mps.synchronize()
    elif device == "cuda":
        import torch

        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCTC, AutoProcessor

    started = time.perf_counter()
    model_path = Path(args.model).expanduser()
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]

    model_load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCTC.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    model = model.to(device=args.device, dtype=dtype).eval()
    synchronize(args.device)
    model_load_seconds = time.perf_counter() - model_load_started

    audio_load_started = time.perf_counter()
    clip = load_audio(args.audio)
    audio_load_seconds = time.perf_counter() - audio_load_started

    frontend_started = time.perf_counter()
    inputs = processor(
        [clip.samples],
        sampling_rate=clip.sample_rate,
        device=args.device,
    )
    inputs = inputs.to(args.device, dtype=model.dtype)
    synchronize(args.device)
    frontend_seconds = time.perf_counter() - frontend_started

    generate_started = time.perf_counter()
    with torch.inference_mode():
        token_ids = model.generate(**inputs)
    synchronize(args.device)
    generate_seconds = time.perf_counter() - generate_started

    decode_started = time.perf_counter()
    text = processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
    decode_seconds = time.perf_counter() - decode_started

    memory_probe_started = time.perf_counter()
    memory = memory_report(args.device)
    memory_probe_seconds = time.perf_counter() - memory_probe_started

    speech_pipeline_seconds = frontend_seconds + generate_seconds + decode_seconds
    result = {
        "schema_version": 1,
        "runtime": "IBM documented Transformers AutoModelForCTC.generate path",
        "source_model_id": args.source_model_id,
        "source_revision": args.source_revision,
        "local_model_path": str(model_path.resolve()),
        "model_file_bytes": (model_path / "model.safetensors").stat().st_size,
        "model_file_sha256": sha256(model_path / "model.safetensors"),
        "audio_file": str(args.audio.resolve()),
        "audio_file_bytes": args.audio.stat().st_size,
        "audio_sha256": sha256(args.audio),
        "audio_duration_seconds": clip.duration,
        "device": args.device,
        "stored_weight_dtype": "bfloat16",
        "runtime_precision": args.precision,
        "model_load_seconds": model_load_seconds,
        "audio_load_seconds": audio_load_seconds,
        "frontend_seconds": frontend_seconds,
        "model_generate_seconds": generate_seconds,
        "decode_seconds": decode_seconds,
        "speech_pipeline_seconds": speech_pipeline_seconds,
        "realtime_multiple": clip.duration / speech_pipeline_seconds,
        "token_frame_count": int(token_ids.shape[-1]),
        "text": text,
        "word_count": len(text.split()),
        "memory": memory,
        "memory_probe_seconds": memory_probe_seconds,
        "elapsed_before_json_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
