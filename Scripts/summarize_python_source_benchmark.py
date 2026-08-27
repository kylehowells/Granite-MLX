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

VARIANTS = (
    {
        "key": "native_bf16_deferred_cast",
        "precision": "bf16",
        "directory": "official-deferred-full-bf16-{index}",
        "transcript": "python-source-bf16.txt",
        "description": "Native source BF16 weights; Granite performs the input cast in its first layer.",
    },
    {
        "key": "fp16_deferred_cast",
        "precision": "fp16",
        "directory": "official-deferred-full-fp16-{index}",
        "transcript": "python-source-fp16.txt",
        "description": "Source BF16 values cast to FP16 at runtime; Granite performs the input cast in its first layer.",
    },
    {
        "key": "promoted_fp32",
        "precision": "fp32",
        "directory": "official-full-fp32-{index}",
        "transcript": "python-source-fp32.txt",
        "description": "Source BF16 values promoted to FP32 at runtime.",
    },
    {
        "key": "native_bf16_documented_precast",
        "precision": "bf16",
        "directory": "official-full-bf16-{index}",
        "transcript": "python-source-bf16-precast.txt",
        "description": "Native BF16 with the current documented pre-generate input cast; affected by MPS silent corruption.",
    },
    {
        "key": "fp16_documented_precast",
        "precision": "fp16",
        "directory": "official-full-fp16-{index}",
        "transcript": "python-source-fp16-precast.txt",
        "description": "Runtime FP16 with the current documented pre-generate input cast; affected by MPS silent corruption.",
    },
)


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
            "runs_per_variant": 3,
            "input_mode": "One unchunked 6,118.72-second waveform",
            "input_cast_finding": (
                "On PyTorch 2.13 MPS, pre-casting the 305,936 x 320 feature tensor and "
                "releasing its FP32 source silently corrupts later computation. Deferring the "
                "cast to Granite's first layer, or retaining the source tensor, is correct."
            ),
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
        "known_mps_issues": [
            {
                "url": "https://github.com/pytorch/pytorch/issues/193487",
                "relevance": "PyTorch 2.13 allocator-state-dependent silent wrong matmul results after large allocations.",
            },
            {
                "url": "https://github.com/pytorch/pytorch/issues/189495",
                "relevance": "Silent biased F.linear corruption when 3D leading dimensions exceed 65,536 rows.",
            },
        ],
        "variants": {},
    }

    args.output.mkdir(parents=True, exist_ok=True)
    variant_transcripts: dict[str, str] = {}
    for variant in VARIANTS:
        precision = variant["precision"]
        runs: list[dict[str, Any]] = []
        transcripts: list[str] = []
        for index in (1, 2, 3):
            run_dir = args.raw_root / variant["directory"].format(index=index)
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
        transcript_path = args.output / variant["transcript"]
        transcript_path.write_text(transcript + "\n")
        variant_transcripts[variant["key"]] = transcript

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
                "Pre-casting the very large feature tensor before generate produced a "
                "deterministic but incomplete decode on M1 Max MPS. Retaining the source "
                "FP32 tensor or deferring the cast makes the same 16-bit runtime complete; "
                "do not use this row as an accuracy or production baseline."
            )
        report["variants"][variant["key"]] = {
            "description": variant["description"],
            "transcript": str(transcript_path.relative_to(args.output.parent.parent)),
            "runs": runs,
            "median": summary,
        }

    fp32_transcript = variant_transcripts["promoted_fp32"]
    for key, transcript in variant_transcripts.items():
        report["variants"][key]["median"]["agreement_vs_python_fp32"] = comparison(
            fp32_transcript, transcript
        )

    retained_dir = args.raw_root / "official-retain-full-fp16-1"
    retained = json.loads((retained_dir / "result.json").read_text())
    retained_transcript = retained.pop("text").strip()
    retained["local_model_path"] = "${SOURCE_MODEL}"
    retained["audio_file"] = "${BENCHMARK_AUDIO}"
    retained["text_sha256"] = digest(retained_transcript)
    retained["agreement_vs_deferred_fp16"] = comparison(
        variant_transcripts["fp16_deferred_cast"], retained_transcript
    )
    report["diagnostics"] = {
        "fp16_documented_precast_while_retaining_fp32_source": retained,
        "finding": (
            "Keeping the otherwise-unused 391,598,080-byte FP32 MPS feature tensor alive "
            "made the explicitly pre-cast FP16 path byte-identical to the deferred-cast FP16 "
            "path. This isolates the failure to MPS allocation/lifetime state rather than "
            "FP16 numerical precision."
        ),
    }

    output_path = args.output / "python-source-results.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
