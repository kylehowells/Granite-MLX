#!/usr/bin/env python3
"""Format a transcript with the source FP32 ONNX punctuation model.

This intentionally uses the model author's ``punctuators`` implementation so
the saved output is the independent baseline for the MLX port.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from punctuators.models import PunctCapSegModelONNX
from punctuators.models.punc_cap_seg_model import PunctCapSegConfigONNX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--batch-size-tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PunctCapSegConfigONNX(
        hf_repo_id=None,
        directory=str(args.model_dir),
        spe_filename="spe_32k_lc_en.model",
        model_filename="punct_cap_seg_en.onnx",
        config_filename="config.yaml",
    )

    load_start = time.perf_counter()
    model = PunctCapSegModelONNX(config, ort_providers=["CPUExecutionProvider"])
    load_seconds = time.perf_counter() - load_start
    source = args.input.read_text(encoding="utf-8").strip()

    inference_start = time.perf_counter()
    sentences = model.infer(
        [source],
        apply_sbd=True,
        batch_size_tokens=args.batch_size_tokens,
        overlap=16,
        num_workers=0,
    )[0]
    inference_seconds = time.perf_counter() - inference_start
    formatted = " ".join(sentences)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(formatted + "\n", encoding="utf-8")
    result = {
        "engine": "onnxruntime-cpu",
        "precision": "fp32",
        "source_transcript": str(args.input),
        "model_directory": str(args.model_dir),
        "output": str(args.output),
        "input_characters": len(source),
        "output_characters": len(formatted),
        "sentence_count": len(sentences),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": load_seconds + inference_seconds,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "batch_size_tokens": args.batch_size_tokens,
        "overlap_tokens": 16,
    }
    args.results.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
