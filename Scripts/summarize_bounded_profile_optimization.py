#!/usr/bin/env python3
"""Summarize bounded-profile and runtime-prototype benchmark rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein


def aggregate(values: list[int | float]) -> dict[str, float]:
    """Return repeat-measurement statistics."""

    numeric = [float(value) for value in values]
    mean = statistics.mean(numeric)
    deviation = statistics.stdev(numeric)
    return {
        "median": statistics.median(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
        "mean": mean,
        "sample_standard_deviation": deviation,
        "coefficient_of_variation_percent": 100 * deviation / mean,
    }


def comparison(reference: str, candidate: str) -> dict[str, Any]:
    """Return transcript agreement diagnostics."""

    reference_words = reference.split()
    candidate_words = candidate.split()
    edits = Levenshtein.distance(reference_words, candidate_words)
    return {
        "reference_words": len(reference_words),
        "candidate_words": len(candidate_words),
        "word_edits": edits,
        "word_agreement_percent": 100 * (1 - edits / len(reference_words)),
        "character_similarity_percent": 100
        * Levenshtein.normalized_similarity(reference, candidate),
    }


def read_run(directory: Path, round_number: int) -> tuple[dict[str, Any], str]:
    """Read one Granite CLI benchmark and external process counters."""

    stderr = (directory / "stderr.txt").read_text()
    transcript = (directory / "transcript.txt").read_text().strip()
    timing = next(json.loads(line) for line in stderr.splitlines() if line.startswith("{"))
    wall = re.search(r"^real ([0-9.]+)$", stderr, re.MULTILINE)
    peak = re.search(r"^\s*(\d+)\s+peak memory footprint$", stderr, re.MULTILINE)
    maximum_resident = re.search(
        r"^\s*(\d+)\s+maximum resident set size$", stderr, re.MULTILINE)
    if not wall or not peak or not maximum_resident:
        raise RuntimeError(f"Missing /usr/bin/time counters in {directory}")
    return {
        "round": round_number,
        "inference_seconds": timing["inference_seconds"],
        "processing_seconds": timing["total_seconds"],
        "process_wall_seconds": float(wall.group(1)),
        "peak_physical_footprint_bytes": int(peak.group(1)),
        "maximum_resident_bytes": int(maximum_resident.group(1)),
        "mlx_peak_memory_bytes": timing["mlx_peak_memory_bytes"],
        "mlx_cache_memory_bytes": timing["mlx_cache_memory_bytes"],
        "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
        "word_count": len(transcript.split()),
    }, transcript


def summarize_group(
    root: Path,
    names: list[str],
    output: Path,
    metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Summarize every named configuration in a raw benchmark group."""

    configurations: dict[str, Any] = {}
    transcripts: dict[str, str] = {}
    for name in names:
        runs: list[dict[str, Any]] = []
        texts: list[str] = []
        for round_number in range(1, 4):
            run, transcript = read_run(root / f"round-{round_number}" / name, round_number)
            runs.append(run)
            texts.append(transcript)
        if len(set(texts)) != 1:
            raise RuntimeError(f"{name} transcript changed across rounds")
        transcript = texts[0]
        transcripts[name] = transcript
        transcript_path = output / "transcripts" / f"{name}.txt"
        transcript_path.write_text(transcript + "\n")
        configurations[name] = {
            **metadata[name],
            "deterministic_across_rounds": True,
            "transcript": str(transcript_path.relative_to(output)),
            "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
            "word_count": len(transcript.split()),
            "runs": runs,
            "summary": {
                key: aggregate([run[key] for run in runs])
                for key in (
                    "inference_seconds",
                    "processing_seconds",
                    "process_wall_seconds",
                    "peak_physical_footprint_bytes",
                    "maximum_resident_bytes",
                    "mlx_peak_memory_bytes",
                    "mlx_cache_memory_bytes",
                )
            },
        }
    return configurations, transcripts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--prototype-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "transcripts").mkdir(exist_ok=True)

    profile_metadata = {
        "current": {"chunk_duration_seconds": 122.88, "context_seconds": 20.48},
        "low-context": {"chunk_duration_seconds": 122.88, "context_seconds": 10.24},
        "same-memory": {"chunk_duration_seconds": 143.36, "context_seconds": 10.24},
    }
    prototype_metadata = {
        "baseline": {"compiled_layers": False, "retain_chunk_cache": False},
        "retained": {"compiled_layers": False, "retain_chunk_cache": True},
        "compiled": {"compiled_layers": True, "retain_chunk_cache": False},
        "both": {"compiled_layers": True, "retain_chunk_cache": True},
    }
    profiles, profile_text = summarize_group(
        args.profile_root, list(profile_metadata), args.output, profile_metadata)
    prototypes, prototype_text = summarize_group(
        args.prototype_root, list(prototype_metadata), args.output, prototype_metadata)
    for name, result in profiles.items():
        result["agreement_vs_current"] = comparison(profile_text["current"], profile_text[name])
    for name, result in prototypes.items():
        result["agreement_vs_baseline"] = comparison(
            prototype_text["baseline"], prototype_text[name])

    current = profiles["current"]["summary"]
    selected = profiles["low-context"]["summary"]
    report = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "audio": {
            "file_name": "granite-cme295-lecture1-16k-mono.wav",
            "duration_seconds": 6118.72,
            "sha256": "f3223fedde7e2212323df3dd59c84193b322fe3c794af60aa9feac7f4044e4ab",
        },
        "methodology": {
            "runs_per_configuration": 3,
            "scheduling": "Rotated interleaving; no configuration ran three times consecutively.",
            "model": "Granite Speech 5.0 TurboCTC MLX mixed-G128/G64 Q8",
            "activation_precision": "FP16 with FP32 softmax stability islands",
            "mlx_cache_limit_mib": 64,
            "system_load": (
                "The profile matrix load average rose from 18.23 to 35.03; "
                "the prototype matrix rose from 25.41 to 64.54 while macOS "
                "media analysis and indexing were active. Use paired relative "
                "results rather than cross-session absolute timing."
            ),
        },
        "chunk_profiles": profiles,
        "runtime_prototypes": prototypes,
        "decision": {
            "selected": "low-context",
            "new_mlx_default_chunk_duration_seconds": 122.88,
            "new_mlx_default_context_seconds": 10.24,
            "inference_speed_improvement_percent": 100 * (
                1 - selected["inference_seconds"]["median"]
                / current["inference_seconds"]["median"]),
            "peak_memory_change_bytes": (
                selected["peak_physical_footprint_bytes"]["median"]
                - current["peak_physical_footprint_bytes"]["median"]),
            "reason": (
                "The low-context profile was materially faster, used less "
                "memory, remained deterministic, and had only 13 word edits "
                "versus the prior default. Compilation and cache retention "
                "did not improve speed without increasing memory."
            ),
        },
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
