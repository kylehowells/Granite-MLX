#!/usr/bin/env python3
"""Benchmark Granite-MLX floating-point and affine quantized checkpoints."""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import sacrebleu
from rapidfuzz.distance import Levenshtein


TIME_PATTERNS = {
    "process_wall_seconds": re.compile(r"^real ([0-9.]+)$", re.MULTILINE),
    "maximum_resident_bytes": re.compile(r"^\s*([0-9]+)\s+maximum resident set size$", re.MULTILINE),
    "peak_memory_footprint_bytes": re.compile(r"^\s*([0-9]+)\s+peak memory footprint$", re.MULTILINE),
}


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def parse_stderr(stderr: str) -> dict[str, Any]:
    timing = next(
        (json.loads(line) for line in stderr.splitlines() if line.startswith("{")),
        None,
    )
    if timing is None:
        raise RuntimeError(f"Missing benchmark JSON in stderr:\n{stderr[-4000:]}")
    for key, pattern in TIME_PATTERNS.items():
        match = pattern.search(stderr)
        if not match:
            raise RuntimeError(f"Missing {key} in time output:\n{stderr[-4000:]}")
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
            f"Benchmark failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}"
        )
    return parse_stderr(completed.stderr), completed.stdout.rstrip("\n")


def median_summary(runs: list[dict[str, Any]]) -> dict[str, float | int]:
    fields = [
        "audio_load_seconds", "model_load_seconds", "inference_seconds",
        "total_seconds", "process_wall_seconds", "real_time_factor",
        "realtime_multiple", "maximum_resident_bytes", "peak_memory_footprint_bytes",
    ]
    result: dict[str, float | int] = {}
    for field in fields:
        value = statistics.median(item[field] for item in runs)
        result[field] = int(value) if field.endswith("_bytes") else value
    return result


def accuracy(reference: str, hypothesis: str) -> dict[str, Any]:
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    word_edits = Levenshtein.distance(reference_words, hypothesis_words)
    return {
        "exact": reference == hypothesis,
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis_words),
        "word_edits": word_edits,
        "word_disagreement_percent": (
            100 * word_edits / len(reference_words) if reference_words else None
        ),
        "character_similarity_percent": 100 * Levenshtein.normalized_similarity(reference, hypothesis),
        "bleu": sacrebleu.corpus_bleu([hypothesis], [[reference]]).score,
        "chrf2": sacrebleu.corpus_chrf([hypothesis], [[reference]]).score,
    }


def hardware() -> dict[str, Any]:
    overview = subprocess.run(
        ["system_profiler", "SPHardwareDataType", "-json"],
        text=True, capture_output=True, check=True,
    )
    profile = json.loads(overview.stdout)["SPHardwareDataType"][0]
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine_name": profile.get("machine_name"),
        "machine_model": profile.get("machine_model"),
        "chip": profile.get("chip_type"),
        "processors": profile.get("number_processors"),
        "physical_memory": profile.get("physical_memory"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--short-audio", type=Path, required=True)
    parser.add_argument("--full-audio", type=Path, required=True)
    parser.add_argument("--python-short-reference", type=Path, required=True)
    parser.add_argument("--python-full-reference-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short-runs", type=int, default=3)
    args = parser.parse_args()

    configurations = [
        ("fp32", args.model_root / "granite-speech-5.0-470m-turboctc-mlx-fp32", None),
        ("fp16", args.model_root / "granite-speech-5.0-470m-turboctc-mlx-fp16", None),
        *[
            (f"q{bits}", args.model_root / f"granite-speech-5.0-470m-turboctc-mlx-q{bits}", bits)
            for bits in (8, 6, 5, 4, 3, 2)
        ],
    ]
    for name, path, _ in configurations:
        if not (path / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing {name} checkpoint: {path}")

    output = args.output.resolve()
    transcript_directory = output.parent / "transcripts"
    transcript_directory.mkdir(parents=True, exist_ok=True)
    python_short = args.python_short_reference.read_text().rstrip("\n")
    python_full = json.loads(args.python_full_reference_json.read_text())["text"].rstrip("\n")

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": hardware(),
        "methodology": {
            "quantization": "Affine weight-only quantization, group size 64; matrix weights quantized, other tensors FP16.",
            "short_runs": args.short_runs,
            "full_runs": 1,
            "memory": "macOS /usr/bin/time -lp maximum resident set size and peak memory footprint.",
            "accuracy": "Transcript disagreement against FP32 references; not ground-truth WER.",
        },
        "audio": {
            "short": str(args.short_audio.resolve()),
            "full": str(args.full_audio.resolve()),
        },
        "references": {
            "python_fp32_short": str(args.python_short_reference.resolve()),
            "python_fp32_full": str(args.python_full_reference_json.resolve()),
        },
        "configurations": {},
    }

    for name, model, bits in configurations:
        print(f"[{name}] warm-up", flush=True)
        run(args.binary, args.short_audio, model)
        short_runs: list[dict[str, Any]] = []
        short_transcript = ""
        for index in range(args.short_runs):
            print(f"[{name}] short {index + 1}/{args.short_runs}", flush=True)
            timing, transcript = run(args.binary, args.short_audio, model)
            short_runs.append(timing)
            if short_transcript and transcript != short_transcript:
                raise RuntimeError(f"Non-deterministic short transcript for {name}")
            short_transcript = transcript

        print(f"[{name}] full 1/1", flush=True)
        full_timing, full_transcript = run(args.binary, args.full_audio, model)
        (transcript_directory / f"{name}-short.txt").write_text(short_transcript + "\n")
        (transcript_directory / f"{name}-full.txt").write_text(full_transcript + "\n")
        entry = {
            "name": name,
            "quantization": None if bits is None else {
                "mode": "affine", "bits": bits, "group_size": 64,
            },
            "model_directory": str(model.resolve()),
            "model_file_bytes": (model / "model.safetensors").stat().st_size,
            "model_directory_bytes": directory_size(model),
            "short": {
                "runs": short_runs,
                "median": median_summary(short_runs),
                "accuracy_vs_python_fp32": accuracy(python_short, short_transcript),
            },
            "full": {
                "runs": [full_timing],
                "median": median_summary([full_timing]),
                "accuracy_vs_python_fp32": accuracy(python_full, full_transcript),
            },
        }
        report["configurations"][name] = entry
        output.write_text(json.dumps(report, indent=2) + "\n")

    swift_short = (transcript_directory / "fp32-short.txt").read_text().rstrip("\n")
    swift_full = (transcript_directory / "fp32-full.txt").read_text().rstrip("\n")
    for entry in report["configurations"].values():
        short = (transcript_directory / f"{entry['name']}-short.txt").read_text().rstrip("\n")
        full = (transcript_directory / f"{entry['name']}-full.txt").read_text().rstrip("\n")
        entry["short"]["accuracy_vs_swift_fp32"] = accuracy(swift_short, short)
        entry["full"]["accuracy_vs_swift_fp32"] = accuracy(swift_full, full)

    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
