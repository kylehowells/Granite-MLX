#!/usr/bin/env python3
"""Build the consolidated Q8 optimization report from preserved run logs."""

from __future__ import annotations

import argparse
import hashlib
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


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text()
    result = json.loads(next(line for line in text.splitlines() if line.startswith("{")))
    for key, pattern in (
        ("maximum_resident_bytes", r"([0-9]+)  maximum resident set size"),
        ("peak_memory_footprint_bytes", r"([0-9]+)  peak memory footprint"),
        ("process_wall_seconds", r"^real ([0-9.]+)$"),
    ):
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            result[key] = int(match.group(1)) if key.endswith("_bytes") else float(match.group(1))
    return result


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "inference_seconds", "realtime_multiple", "maximum_resident_bytes",
        "peak_memory_footprint_bytes", "process_wall_seconds",
    )
    median = {
        field: statistics.median(run[field] for run in runs if field in run)
        for field in fields
    }
    return {"runs": runs, "median": median}


def accuracy(reference: str, candidate: str) -> dict[str, Any]:
    reference_words = reference.split()
    candidate_words = candidate.split()
    edits = Levenshtein.distance(reference_words, candidate_words)
    return {
        "reference_words": len(reference_words),
        "candidate_words": len(candidate_words),
        "word_edits": edits,
        "word_disagreement_percent": 100 * edits / len(reference_words),
        "character_similarity_percent": 100 * Levenshtein.normalized_similarity(reference, candidate),
        "bleu": sacrebleu.corpus_bleu([candidate], [[reference]]).score,
        "chrf2": sacrebleu.corpus_chrf([candidate], [[reference]]).score,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/tmp/granite-q8-optimization"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_root
    repo = Path(__file__).resolve().parents[1]
    model_root = Path("/Users/kylehowells/Developer/ML-Models")
    full_audio = Path("/private/tmp/granite-cme295-lecture1-16k-mono.wav")
    qmm = json.loads((repo / "Benchmarks/q8-optimization/qmm-results.json").read_text())

    fp32_reference = (repo / "Benchmarks/quantization/transcripts/fp32-full.txt").read_text().strip()
    paired: dict[str, Any] = {}
    for name, variant in (("q8_g64_fp16", "q8"), ("q8_g128_fp16", "q8-g128")):
        runs = [parse_log(raw / f"interleaved/{variant}-{index}.log") for index in (1, 2, 3)]
        transcripts = [(raw / f"interleaved/{variant}-{index}.txt").read_text().strip() for index in (1, 2, 3)]
        entry = summarize(runs)
        entry["transcripts_identical"] = len(set(transcripts)) == 1
        entry["accuracy_vs_swift_fp32_weights"] = accuracy(fp32_reference, transcripts[0])
        paired[name] = entry

    experiments = {}
    experiment_specs = {
        "q8_g128_fp8_emulated": (
            raw / "experiment-paired/fp8.log",
            raw / "experiment-paired/fp8.txt",
            raw / "experiment-paired/fp16-before-fp8.log",
            raw / "experiment-paired/fp16-before-fp8.txt",
        ),
        "q8_g128_int8_emulated": (
            raw / "experiment-paired/int8.log",
            raw / "experiment-paired/int8.txt",
            raw / "experiment-paired/fp16-before-int8.log",
            raw / "experiment-paired/fp16-before-int8.txt",
        ),
        "q8_g128_fp16_ctc_tile_2048": (
            raw / "experiment-paired/ctc2048.log",
            raw / "experiment-paired/ctc2048.txt",
            raw / "experiment-paired/fp16-before-ctc.log",
            raw / "experiment-paired/fp16-before-ctc.txt",
        ),
    }
    for name, (log, transcript_path, control_log, control_transcript_path) in experiment_specs.items():
        transcript = transcript_path.read_text().strip()
        control_transcript = control_transcript_path.read_text().strip()
        candidate_run = parse_log(log)
        control_run = parse_log(control_log)
        experiments[name] = {
            **summarize([candidate_run]),
            "paired_fp16_control": summarize([control_run]),
            "inference_time_ratio_vs_paired_control": (
                candidate_run["inference_seconds"] / control_run["inference_seconds"]
            ),
            "peak_memory_ratio_vs_paired_control": (
                candidate_run["peak_memory_footprint_bytes"]
                / control_run["peak_memory_footprint_bytes"]
            ),
            "accuracy_vs_q8_g128_fp16": accuracy(control_transcript, transcript),
        }

    q64_model = model_root / "granite-speech-5.0-470m-turboctc-mlx-q8/model.safetensors"
    q128_model = model_root / "granite-speech-5.0-470m-turboctc-mlx-q8-g128/model.safetensors"
    hardware = subprocess.run(
        ["system_profiler", "SPHardwareDataType", "-json"],
        text=True, capture_output=True, check=True,
    )
    hardware = json.loads(hardware.stdout)["SPHardwareDataType"][0]
    report = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx_python": qmm["environment"]["mlx_version"],
            "mlx_swift": "0.31.4",
            "device": qmm["environment"]["device"],
            "hardware": hardware,
        },
        "audio": {
            "path": str(full_audio),
            "sha256": sha256(full_audio),
            "duration_seconds": 6118.72,
        },
        "model_files": {
            "q8_g64_bytes": q64_model.stat().st_size,
            "q8_g128_mixed_bytes": q128_model.stat().st_size,
            "q8_g128_override": {"encoder.input_linear": 64},
        },
        "paired_full_input_results": paired,
        "experimental_full_input_results": experiments,
        "capabilities": qmm["capabilities"],
        "artifacts": {
            "qmm_microbenchmarks": "qmm-results.json",
            "dtype_audit": "dtype-audit-fp16.json",
            "metal_trace_toc": "metal/trace-toc.xml",
            "metal_shader_list": "metal/shaders.xml",
        },
        "kernel_experiments": {
            "symmetric_signed_q8_g128": {
                "status": "rejected",
                "reason": "Numerically valid scalar/threadgroup prototype was 2x to 13.5x slower than MLX affine QMM.",
            },
            "mlx_qmm_bm64": {
                "status": "rejected",
                "reason": "Experimental 64-frame tile failed transcript correctness and did not outperform the stock valid kernel.",
            },
            "fused_ctc_qmm_argmax": {
                "status": "blocked_by_public_api",
                "implemented_fallback": "Exact streamed vocabulary-tile QMM plus running argmax.",
                "result": "Fallback was exact but slower and did not reduce peak because middle CTC logits establish the peak.",
                "required_work": "A new MLX primitive/Metal QMM epilogue that emits per-tile maxima and IDs instead of logits.",
            },
        },
        "conclusion": {
            "recommended_weights": "Q8 affine, default group 128 with encoder.input_linear group 64",
            "recommended_activations": "FP16 with FP32 softmax stability islands",
            "native_fp8_or_int8_activation_compute_on_m1_max": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
