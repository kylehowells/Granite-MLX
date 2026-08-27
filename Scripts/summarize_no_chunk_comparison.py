#!/usr/bin/env python3
"""Summarize paired Swift MLX and Python MPS no-chunk benchmarks."""

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


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def external_time(path: Path) -> dict[str, Any]:
    text = path.read_text()
    patterns = {
        "process_wall_seconds": (r"^real ([0-9.]+)$", float),
        "process_user_seconds": (r"^user ([0-9.]+)$", float),
        "process_system_seconds": (r"^sys ([0-9.]+)$", float),
        "maximum_resident_bytes": (
            r"^\s*(\d+)\s+maximum resident set size$",
            int,
        ),
        "peak_physical_footprint_bytes": (
            r"^\s*(\d+)\s+peak memory footprint$",
            int,
        ),
    }
    result: dict[str, Any] = {}
    for key, (pattern, cast) in patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            raise RuntimeError(f"Missing {key} in {path}")
        result[key] = cast(match.group(1))
    return result


def benchmark_json(path: Path) -> dict[str, Any]:
    return next(json.loads(line) for line in path.read_text().splitlines() if line.startswith("{"))


def comparison(reference: str, candidate: str) -> dict[str, Any]:
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


def median(rows: list[dict[str, Any]], key: str) -> int | float:
    return statistics.median(row[key] for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    runtimes: dict[str, dict[str, Any]] = {}
    transcripts: dict[str, str] = {}
    for runtime in ("swift", "python"):
        rows: list[dict[str, Any]] = []
        texts: list[str] = []
        for pair in (1, 2, 3):
            run_dir = args.raw_root / "paired" / f"{pair}-{runtime}"
            external = external_time(run_dir / "stderr.txt")
            if runtime == "swift":
                timing = benchmark_json(run_dir / "stderr.txt")
                text = (run_dir / "transcript.txt").read_text().strip()
                row = {
                    "pair": pair,
                    "speech_seconds": timing["inference_seconds"],
                    "processing_seconds": timing["total_seconds"],
                    "model_load_seconds": timing["model_load_seconds"],
                    "audio_load_seconds": timing["audio_load_seconds"],
                    "mlx_peak_memory_bytes": timing["mlx_peak_memory_bytes"],
                    **external,
                }
            else:
                timing = json.loads((run_dir / "result.json").read_text())
                text = timing.pop("text").strip()
                row = {
                    "pair": pair,
                    "speech_seconds": timing["speech_pipeline_seconds"],
                    "processing_seconds": timing["elapsed_before_json_seconds"],
                    "model_load_seconds": timing["model_load_seconds"],
                    "audio_load_seconds": timing["audio_load_seconds"],
                    "mps_driver_allocated_bytes": timing["memory"]["mps_driver_allocated_bytes"],
                    **external,
                    "peak_physical_footprint_bytes": timing["memory"][
                        "peak_physical_footprint_bytes"
                    ],
                }
            row["text_sha256"] = digest(text)
            rows.append(row)
            texts.append(text)

        if len(set(texts)) != 1:
            raise RuntimeError(f"{runtime} transcript changed across paired runs")
        transcripts[runtime] = texts[0]
        transcript_path = args.output / f"{runtime}-fp16.txt"
        transcript_path.write_text(texts[0] + "\n")
        summary_keys = (
            "speech_seconds",
            "processing_seconds",
            "process_wall_seconds",
            "maximum_resident_bytes",
            "peak_physical_footprint_bytes",
        )
        runtimes[runtime] = {
            "runs": rows,
            "median": {key: median(rows, key) for key in summary_keys},
            "transcript": str(transcript_path.name),
            "word_count": len(texts[0].split()),
        }

    swift_seconds = runtimes["swift"]["median"]["speech_seconds"]
    python_seconds = runtimes["python"]["median"]["speech_seconds"]
    report = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "audio": {
            "duration_seconds": 6118.72,
            "sha256": "f3223fedde7e2212323df3dd59c84193b322fe3c794af60aa9feac7f4044e4ab",
        },
        "methodology": {
            "runs": "Three paired runs with alternating order under the same sustained-load session.",
            "swift": "Native Swift MLX, converted FP16 weights, FP16 activations, no chunking, no formatter.",
            "python": "PyTorch MPS, source BF16 values cast to FP16, deferred input cast, no chunking.",
            "caution": "Absolute one-pass time varied materially with session load; use the paired ratio for backend comparison.",
        },
        "runtimes": runtimes,
        "comparison": {
            "swift_to_python_speech_time_ratio": swift_seconds / python_seconds,
            "swift_speed_advantage_percent": 100 * (1 - swift_seconds / python_seconds),
            "output": comparison(transcripts["python"], transcripts["swift"]),
        },
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
