#!/usr/bin/env python3
"""Summarize interleaved Python, MLX, and Core ML benchmark rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein


TIME_PATTERNS = {
    "process_wall_seconds": re.compile(r"^real ([0-9.]+)$", re.MULTILINE),
    "maximum_resident_bytes": re.compile(
        r"^\s*(\d+)\s+maximum resident set size$", re.MULTILINE
    ),
    "peak_physical_footprint_bytes": re.compile(
        r"^\s*(\d+)\s+peak memory footprint$", re.MULTILINE
    ),
}


def sha256_text(text: str) -> str:
    """Return a stable digest for transcript text."""

    return hashlib.sha256(text.encode()).hexdigest()


def parse_external_time(stderr: str) -> dict[str, int | float]:
    """Extract macOS `/usr/bin/time -lp` counters."""

    result: dict[str, int | float] = {}
    for key, pattern in TIME_PATTERNS.items():
        match = pattern.search(stderr)
        if not match:
            raise RuntimeError(f"Missing {key} in timed process output")
        result[key] = float(match.group(1)) if key == "process_wall_seconds" else int(match.group(1))
    return result


def cli_timing(stderr: str) -> dict[str, Any]:
    """Read the machine-readable Granite CLI timing line."""

    for line in stderr.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError("Missing Granite CLI benchmark JSON")


def normalize_lexical(text: str) -> list[str]:
    """Remove case and punctuation while preserving letters and numbers."""

    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = "".join(
        character if character.isalnum() or character in "' " else " "
        for character in normalized
    )
    return " ".join(normalized.split()).split()


def comparison(reference: str, candidate: str) -> dict[str, Any]:
    """Compare two transcripts without presenting the result as human WER."""

    reference_words = reference.split()
    candidate_words = candidate.split()
    edits = Levenshtein.distance(reference_words, candidate_words)
    lexical_reference = normalize_lexical(reference)
    lexical_candidate = normalize_lexical(candidate)
    lexical_edits = Levenshtein.distance(lexical_reference, lexical_candidate)
    return {
        "reference_words": len(reference_words),
        "candidate_words": len(candidate_words),
        "word_edits": edits,
        "word_agreement_percent": 100 * (1 - edits / len(reference_words)),
        "character_similarity_percent": 100
        * Levenshtein.normalized_similarity(reference, candidate),
        "lexical_word_edits": lexical_edits,
        "lexical_word_agreement_percent": 100
        * (1 - lexical_edits / len(lexical_reference)),
    }


def aggregate(values: list[int | float]) -> dict[str, float]:
    """Return distribution statistics for three repeated measurements."""

    numeric = [float(value) for value in values]
    median = statistics.median(numeric)
    mean = statistics.mean(numeric)
    standard_deviation = statistics.stdev(numeric)
    return {
        "median": median,
        "minimum": min(numeric),
        "maximum": max(numeric),
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "coefficient_of_variation_percent": (
            100 * standard_deviation / mean if mean else 0
        ),
        "range_relative_to_median_percent": (
            100 * (max(numeric) - min(numeric)) / median if median else 0
        ),
    }


def sanitize_model(value: str) -> str:
    """Replace machine-specific model paths with portable artifact names."""

    path = Path(value)
    return path.name if path.is_absolute() else value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    transcript_directory = args.output / "transcripts"
    transcript_directory.mkdir(exist_ok=True)

    manifest = json.loads((args.raw_root / "manifest.json").read_text())
    configurations: dict[str, Any] = {}
    transcripts: dict[str, str] = {}
    python_environment: dict[str, str] | None = None

    for name in manifest["order"]:
        runs: list[dict[str, Any]] = []
        texts: list[str] = []
        artifact_bytes: int | None = None
        for round_number in range(1, manifest["rounds"] + 1):
            directory = args.raw_root / f"round-{round_number}" / name
            metadata = json.loads((directory / "run.json").read_text())
            stdout = (directory / "stdout.txt").read_text().strip()
            stderr = (directory / "stderr.txt").read_text()
            external = parse_external_time(stderr)
            if name.startswith("python-"):
                timing = json.loads(stdout)
                text = timing.pop("text").strip()
                python_environment = timing["environment"]
                artifact_bytes = timing["model_file_bytes"]
                peak = timing["memory"]["peak_physical_footprint_bytes"]
                row = {
                    "round": round_number,
                    "speech_inference_seconds": timing["speech_pipeline_seconds"],
                    "processing_seconds": timing["elapsed_before_json_seconds"],
                    "model_load_seconds": timing["model_load_seconds"],
                    "audio_load_seconds": timing["audio_load_seconds"],
                    "frontend_seconds": timing["frontend_seconds"],
                    "model_generate_seconds": timing["model_generate_seconds"],
                    "decode_seconds": timing["decode_seconds"],
                    "maximum_resident_bytes": external["maximum_resident_bytes"],
                    "peak_physical_footprint_bytes": peak,
                    "mps_driver_allocated_bytes": timing["memory"]["mps_driver_allocated_bytes"],
                    "process_wall_seconds": external["process_wall_seconds"],
                }
            else:
                timing = cli_timing(stderr)
                text = stdout
                model = Path(timing["model"])
                weights = model / "model.safetensors"
                if weights.is_file():
                    artifact_bytes = weights.stat().st_size
                elif model.suffix == ".mlpackage":
                    artifact_bytes = sum(path.stat().st_size for path in model.rglob("*") if path.is_file())
                row = {
                    "round": round_number,
                    "speech_inference_seconds": timing["inference_seconds"],
                    "processing_seconds": timing["total_seconds"],
                    "model_load_seconds": timing["model_load_seconds"],
                    "audio_load_seconds": timing["audio_load_seconds"],
                    "punctuation_model_load_seconds": timing["punctuation_model_load_seconds"],
                    "punctuation_inference_seconds": timing["punctuation_inference_seconds"],
                    "maximum_resident_bytes": external["maximum_resident_bytes"],
                    "peak_physical_footprint_bytes": external["peak_physical_footprint_bytes"],
                    "process_wall_seconds": external["process_wall_seconds"],
                    "backend": timing["backend"],
                    "model": sanitize_model(timing["model"]),
                    "chunk_duration_seconds": timing["audio_chunk_duration_seconds"],
                    "chunk_context_seconds": timing["audio_chunk_context_seconds"],
                }
                for key in (
                    "coreml_chunk_count",
                    "coreml_feature_extraction_seconds",
                    "coreml_input_copy_seconds",
                    "coreml_output_copy_seconds",
                    "coreml_prediction_seconds",
                    "mlx_peak_memory_bytes",
                ):
                    if key in timing:
                        row[key] = timing[key]
            row["machine_before"] = metadata["machine_before"]
            row["machine_after"] = metadata["machine_after"]
            row["transcript_sha256"] = sha256_text(text)
            row["word_count"] = len(text.split())
            runs.append(row)
            texts.append(text)

        if len(set(texts)) != 1:
            raise RuntimeError(f"{name} transcript changed across rounds")
        transcript = texts[0]
        transcripts[name] = transcript
        transcript_path = transcript_directory / f"{name}.txt"
        transcript_path.write_text(transcript + "\n")
        metrics = {}
        for key in (
            "speech_inference_seconds",
            "processing_seconds",
            "model_load_seconds",
            "audio_load_seconds",
            "punctuation_model_load_seconds",
            "punctuation_inference_seconds",
            "maximum_resident_bytes",
            "peak_physical_footprint_bytes",
            "process_wall_seconds",
        ):
            values = [row[key] for row in runs if key in row]
            if values:
                metrics[key] = aggregate(values)
        configurations[name] = {
            "artifact_bytes": artifact_bytes,
            "transcript": str(transcript_path.relative_to(args.output)),
            "transcript_sha256": sha256_text(transcript),
            "word_count": len(transcript.split()),
            "deterministic_across_rounds": True,
            "runs": runs,
            "summary": metrics,
        }

    raw_reference = transcripts["swift-source-bounded"]
    for name in (
        "python-bf16-one-pass",
        "python-fp16-one-pass",
        "python-fp32-one-pass",
        "swift-source-bounded",
        "swift-q8-bounded",
        "swift-fp16-one-pass",
        "coreml-q8-bounded",
    ):
        configurations[name]["agreement_vs_swift_source"] = comparison(
            raw_reference, transcripts[name]
        )
    fp32_reference = transcripts["python-fp32-one-pass"]
    for name in ("python-bf16-one-pass", "python-fp16-one-pass", "python-fp32-one-pass"):
        configurations[name]["agreement_vs_python_fp32"] = comparison(
            fp32_reference, transcripts[name]
        )

    formatted_comparison = comparison(
        transcripts["swift-q8-formatted"], transcripts["coreml-q8-formatted"]
    )
    report = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "methodology": {
            "rounds": manifest["rounds"],
            "order": manifest["order"],
            "scheduling": "Every configuration ran once per round in the same interleaved order; no configuration ran three times consecutively.",
            "statistics": "Tables use the median of three clean processes. JSON retains every run, min/max, sample standard deviation, and coefficient of variation.",
            "speech_timing": "Python: frontend + model.generate + decode. Swift: speech recognizer inference, excluding model/audio loading and punctuation.",
            "wall_timing": "macOS /usr/bin/time -lp around the complete process.",
            "memory": "Peak physical footprint from macOS process accounting; Python also records the MPS driver allocator.",
            "agreement": "Levenshtein agreement against native Swift source output, not WER against a human transcript.",
        },
        "system": {
            "host": manifest["host"],
            "python": python_environment,
        },
        "audio": {
            "file_name": Path(manifest["audio"]["path"]).name,
            "bytes": manifest["audio"]["bytes"],
            "sha256": manifest["audio"]["sha256"],
            "duration_seconds": 6118.72,
            "description": "Stanford CME295 lecture, mono 16 kHz",
        },
        "configurations": configurations,
        "comparisons": {
            "coreml_vs_mlx_raw": comparison(
                transcripts["swift-q8-bounded"], transcripts["coreml-q8-bounded"]
            ),
            "coreml_vs_mlx_formatted": formatted_comparison,
        },
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
