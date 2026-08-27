#!/usr/bin/env python3
"""Benchmark an MLX punctuation checkpoint and compare with ONNX FP32 text."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from Scripts.punctuation_mlx import MLXPunctuationModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    return parser.parse_args()


def disagreement(reference: str, candidate: str) -> dict[str, float | int]:
    ref_words = reference.split()
    cand_words = candidate.split()
    return {
        "character_edit_distance": Levenshtein.distance(reference, candidate),
        "character_disagreement_percent": 100 * Levenshtein.normalized_distance(reference, candidate),
        "word_edit_distance": Levenshtein.distance(ref_words, cand_words),
        "word_disagreement_percent": 100 * Levenshtein.normalized_distance(ref_words, cand_words),
        "exact_match": reference == candidate,
    }


def main() -> None:
    args = parse_args()
    source = args.input.read_text(encoding="utf-8").strip()
    baseline = args.baseline.read_text(encoding="utf-8").strip()
    load_start = time.perf_counter()
    model = MLXPunctuationModel(args.model_dir)
    load_seconds = time.perf_counter() - load_start
    start = time.perf_counter()
    sentences = model.infer([source])[0]
    inference_seconds = time.perf_counter() - start
    formatted = " ".join(sentences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(formatted + "\n", encoding="utf-8")
    config = model.config
    result = {
        "engine": "mlx",
        "precision": config["precision"],
        "quantization": config["quantization"],
        "model_directory": str(args.model_dir),
        "model_bytes": sum(p.stat().st_size for p in args.model_dir.rglob("*") if p.is_file()),
        "output": str(args.output),
        "sentence_count": len(sentences),
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": load_seconds + inference_seconds,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        **disagreement(baseline, formatted),
    }
    args.results.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
