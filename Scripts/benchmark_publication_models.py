#!/usr/bin/env python3
"""Benchmark the two Granite Speech source models and publication checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import sacrebleu
from rapidfuzz.distance import Levenshtein


TIME_PATTERNS = {
    "process_wall_seconds": re.compile(r"^real ([0-9.]+)$", re.MULTILINE),
    "maximum_resident_bytes": re.compile(
        r"^\s*([0-9]+)\s+maximum resident set size$", re.MULTILINE
    ),
    "peak_memory_footprint_bytes": re.compile(
        r"^\s*([0-9]+)\s+peak memory footprint$", re.MULTILINE
    ),
}

FAMILIES = {
    "apache": {
        "source_model_id": "ibm-granite/granite-speech-5.0-470m-turboctc",
        "source_revision": "7e74c6438b7cfb5090cb6a131538f5e8515a7de3",
        "source_directory": "granite-speech-5.0-470m-turboctc",
        "converted_prefix": "granite-speech-5.0-470m-turboctc-mlx-",
        "variant_directories": {"q8": "granite-speech-5.0-470m-turboctc-mlx-q8-g128"},
    },
    "non_commercial": {
        "source_model_id": "ibm-granite/granite-speech-5.0-470m-turboctc-nc",
        "source_revision": "0eb7b4fe726a294815dc45d342860465b5af68ef",
        "source_directory": "granite-speech-5.0-470m-turboctc-nc",
        "converted_prefix": "granite-speech-5.0-470m-turboctc-nc-mlx-",
        "variant_directories": {},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_stderr(stderr: str) -> dict[str, Any]:
    timing = next(
        (json.loads(line) for line in stderr.splitlines() if line.startswith("{")),
        None,
    )
    if timing is None:
        raise RuntimeError(f"Missing benchmark JSON:\n{stderr[-4000:]}")
    for key, pattern in TIME_PATTERNS.items():
        match = pattern.search(stderr)
        if not match:
            raise RuntimeError(f"Missing {key}:\n{stderr[-4000:]}")
        timing[key] = float(match.group(1)) if key == "process_wall_seconds" else int(match.group(1))
    return timing


def run(binary: Path, audio: Path, model: Path) -> tuple[dict[str, Any], str]:
    command = [
        "/usr/bin/time", "-lp", str(binary), str(audio),
        "--model", str(model), "--benchmark",
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"Benchmark failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr[-4000:]}"
        )
    return parse_stderr(completed.stderr), completed.stdout.rstrip("\n")


def comparison(reference: str, hypothesis: str) -> dict[str, Any]:
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    edits = Levenshtein.distance(reference_words, hypothesis_words)
    disagreement = 100 * edits / len(reference_words) if reference_words else None
    return {
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis_words),
        "word_edits": edits,
        "word_disagreement_percent": disagreement,
        "word_agreement_percent": None if disagreement is None else 100 - disagreement,
        "character_similarity_percent": 100 * Levenshtein.normalized_similarity(reference, hypothesis),
        "bleu": sacrebleu.corpus_bleu([hypothesis], [[reference]]).score,
        "chrf2": sacrebleu.corpus_chrf([hypothesis], [[reference]]).score,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse rows whose model path is unchanged")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.json"
    previous_report = (
        json.loads(results_path.read_text())
        if args.resume and results_path.is_file()
        else {}
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": {"platform": platform.platform(), "machine": platform.machine()},
        "methodology": {
            "audio_duration_seconds": 6118.72,
            "runtime": "Native Swift Granite-MLX release build",
            "activation_precision": "FP16",
            "chunk_duration_seconds": 122.88,
            "chunk_context_seconds": 20.48,
            "mlx_cache_limit_mib": 64,
            "runs_per_checkpoint": 1,
            "accuracy": (
                "Transcript agreement with the matching original IBM source weights on the "
                "same audio and runtime; this is not ground-truth WER."
            ),
        },
        "audio": {
            "path": str(args.audio.resolve()),
            "sha256": sha256(args.audio),
            "bytes": args.audio.stat().st_size,
        },
        "families": {},
    }

    for family_name, metadata in FAMILIES.items():
        previous_family = previous_report.get("families", {}).get(family_name)
        family: dict[str, Any] = {**metadata, "checkpoints": {}}
        report["families"][family_name] = family
        variants = ["source", "fp16", "q8", "q6", "q5", "q4"]
        for variant in variants:
            directory_name = metadata["variant_directories"].get(variant) or (
                metadata["source_directory"] if variant == "source"
                else metadata["converted_prefix"] + variant
            )
            model = args.model_root / directory_name
            weights = model / "model.safetensors"
            if not weights.is_file():
                raise FileNotFoundError(weights)
            transcript_path = args.output / f"{family_name}-{variant}.txt"
            previous = (previous_family or {}).get("checkpoints", {}).get(variant)
            if (
                args.resume and previous
                and previous.get("model_directory") == str(model.resolve())
                and transcript_path.is_file()
            ):
                print(f"[{family_name}/{variant}] reusing existing result", flush=True)
                family["checkpoints"][variant] = previous
            else:
                print(f"[{family_name}/{variant}] full lecture", flush=True)
                timing, transcript = run(args.binary, args.audio, model)
                transcript_path.write_text(transcript + "\n")
                family["checkpoints"][variant] = {
                    "model_directory": str(model.resolve()),
                    "model_file_bytes": weights.stat().st_size,
                    "model_file_sha256": sha256(weights),
                    "timing": timing,
                    "transcript": str(transcript_path.resolve()),
                }
            results_path.write_text(json.dumps(report, indent=2) + "\n")

        reference = args.output.joinpath(f"{family_name}-source.txt").read_text().rstrip("\n")
        for variant, checkpoint in family["checkpoints"].items():
            hypothesis = args.output.joinpath(f"{family_name}-{variant}.txt").read_text().rstrip("\n")
            checkpoint["accuracy_vs_source"] = comparison(reference, hypothesis)
        results_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.output / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
