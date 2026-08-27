#!/usr/bin/env python3
"""Summarize repeated IBM source-weight Python benchmarks into checked-in JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import sacrebleu
from rapidfuzz.distance import Levenshtein


TIME_PATTERNS = {
    "process_wall_seconds": (re.compile(r"^real ([0-9.]+)$", re.MULTILINE), float),
    "process_user_seconds": (re.compile(r"^user ([0-9.]+)$", re.MULTILINE), float),
    "process_system_seconds": (re.compile(r"^sys ([0-9.]+)$", re.MULTILINE), float),
    "maximum_resident_bytes": (
        re.compile(r"^\s*(\d+)\s+maximum resident set size$", re.MULTILINE),
        int,
    ),
}


def comparison(reference: str, hypothesis: str) -> dict[str, Any]:
    """Return output-agreement diagnostics; these are not ground-truth WER."""

    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    edits = Levenshtein.distance(reference_words, hypothesis_words)
    disagreement = 100 * edits / len(reference_words)
    return {
        "reference_words": len(reference_words),
        "hypothesis_words": len(hypothesis_words),
        "word_edits": edits,
        "word_disagreement_percent": disagreement,
        "word_agreement_percent": 100 - disagreement,
        "character_similarity_percent": 100
        * Levenshtein.normalized_similarity(reference, hypothesis),
        "bleu": sacrebleu.corpus_bleu([hypothesis], [[reference]]).score,
        "chrf2": sacrebleu.corpus_chrf([hypothesis], [[reference]]).score,
    }


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def command(*arguments: str) -> str:
    return subprocess.run(arguments, check=True, capture_output=True, text=True).stdout.strip()


def median(values: list[int | float]) -> int | float:
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = args.reference.read_text().strip()
    repository_root = Path(__file__).resolve().parent.parent
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "system": {
            "hardware": command("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"),
            "physical_memory_bytes": int(
                command("/usr/sbin/sysctl", "-n", "hw.memsize")
            ),
            "os_version": command("/usr/bin/sw_vers", "-productVersion"),
            "os_build": command("/usr/bin/sw_vers", "-buildVersion"),
        },
        "reproducibility": {
            "benchmark_script": "Scripts/benchmark_python_source.py",
            "benchmark_script_sha256": file_digest(
                repository_root / "Scripts/benchmark_python_source.py"
            ),
            "reference_support_script": "Scripts/granite_reference.py",
            "reference_support_script_sha256": file_digest(
                repository_root / "Scripts/granite_reference.py"
            ),
        },
        "methodology": {
            "runtime": "IBM documented Transformers AutoModelForCTC.generate path",
            "device": "MPS",
            "runs_per_precision": 3,
            "input_mode": "One unchunked 6,118.72-second waveform",
            "timing": (
                "Speech pipeline is frontend + model.generate + tokenizer decode. Process wall "
                "is /usr/bin/time -lp and includes startup, model/audio loading, a post-run "
                "memory probe, and JSON output."
            ),
            "memory": (
                "Peak physical footprint comes from macOS /usr/bin/footprint. Maximum RSS and "
                "PyTorch MPS allocator values are retained separately and are not summed."
            ),
            "agreement": (
                "Word agreement is against the native Swift source-weight output on the same "
                "audio. Backend precision and one-pass versus bounded chunking also affect it; "
                "this is not WER against a human transcript."
            ),
        },
        "precisions": {},
    }

    args.output.mkdir(parents=True, exist_ok=True)
    for precision in ("bf16", "fp32"):
        runs: list[dict[str, Any]] = []
        transcripts: list[str] = []
        for index in (1, 2, 3):
            run_dir = args.raw_root / f"official-full-{precision}-{index}"
            raw = json.loads((run_dir / "result.json").read_text())
            transcript = raw.pop("text").strip()
            transcripts.append(transcript)
            raw["local_model_path"] = "${SOURCE_MODEL}"
            raw["audio_file"] = "${BENCHMARK_AUDIO}"
            raw["text_sha256"] = digest(transcript)

            time_output = (run_dir / "time.txt").read_text()
            external: dict[str, Any] = {}
            for key, (pattern, cast) in TIME_PATTERNS.items():
                match = pattern.search(time_output)
                if not match:
                    raise RuntimeError(f"Missing {key} in {run_dir / 'time.txt'}")
                external[key] = cast(match.group(1))
            raw["external_process"] = external
            runs.append(raw)

        if len(set(transcripts)) != 1:
            raise RuntimeError(f"{precision} transcript was not deterministic across runs")
        transcript = transcripts[0]
        transcript_path = args.output / f"python-source-{precision}.txt"
        transcript_path.write_text(transcript + "\n")

        timing_keys = (
            "model_load_seconds",
            "audio_load_seconds",
            "frontend_seconds",
            "model_generate_seconds",
            "decode_seconds",
            "speech_pipeline_seconds",
            "realtime_multiple",
        )
        memory_keys = (
            "maximum_resident_bytes",
            "mps_current_allocated_bytes",
            "mps_driver_allocated_bytes",
            "physical_footprint_bytes",
            "peak_physical_footprint_bytes",
        )
        summary = {key: median([run[key] for run in runs]) for key in timing_keys}
        summary.update(
            {
                key: median([run["memory"][key] for run in runs])
                for key in memory_keys
            }
        )
        summary.update(
            {
                key: median([run["external_process"][key] for run in runs])
                for key in TIME_PATTERNS
            }
        )
        summary["text_sha256"] = digest(transcript)
        summary["word_count"] = len(transcript.split())
        summary["agreement_vs_swift_source"] = comparison(reference, transcript)

        complete = summary["word_count"] >= 0.9 * len(reference.split())
        summary["complete_long_form_decode"] = complete
        if not complete:
            summary["quality_note"] = (
                "The documented model-dtype BF16 input cast produced a deterministic but "
                "incomplete long-form decode on M1 Max MPS; do not use this row as an "
                "accuracy or production baseline."
            )
        report["precisions"][precision] = {
            "transcript": str(transcript_path.relative_to(args.output.parent.parent)),
            "runs": runs,
            "median": summary,
        }

    output_path = args.output / "python-source-results.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
