#!/usr/bin/env python3
"""Consolidate Granite long-input memory experiments into JSON."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from rapidfuzz.distance import Levenshtein


def parse_run(root: Path, name: str) -> dict:
    log = (root / f"{name}.log").read_text()
    result = json.loads(next(line for line in log.splitlines() if line.startswith("{")))
    for key, pattern in (
        ("maximum_resident_bytes", r"([0-9]+)  maximum resident set size"),
        ("peak_memory_footprint_bytes", r"([0-9]+)  peak memory footprint"),
        ("process_wall_seconds", r"^real ([0-9.]+)$"),
    ):
        match = re.search(pattern, log, re.MULTILINE)
        if match:
            result[key] = int(match.group(1)) if key.endswith("_bytes") else float(match.group(1))
    result["transcript_path"] = f"raw:{root / f'{name}.txt'}"
    return result


def disagreement(reference: str, candidate: str) -> dict:
    reference_words = reference.split()
    candidate_words = candidate.split()
    edits = Levenshtein.distance(reference_words, candidate_words)
    return {
        "reference_words": len(reference_words),
        "candidate_words": len(candidate_words),
        "word_edits": edits,
        "word_disagreement_percent": 100 * edits / len(reference_words),
        "character_similarity_percent": 100 * Levenshtein.normalized_similarity(reference, candidate),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/tmp/granite-memory-experiments/full"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.raw_root

    names = [
        "baseline-default", "baseline-cache0", "baseline-cache64",
        "middle4096-cache0", "middle2048-cache0", "middle2048-final2048-cache0",
        "chunk122.88", "chunk30.72", "chunk10.24",
        "chunk122.88-context10.24", "chunk122.88-context20.48",
        "chunk122.88-context20.48-cache64",
        "q6-onepass-cache0", "q6-chunk122-context20",
        "q5-onepass-cache0", "q5-chunk122-context20",
    ]
    runs = {name: parse_run(root, name) for name in names}
    transcript_directory = args.output.parent / "transcripts"
    transcript_directory.mkdir(parents=True, exist_ok=True)
    for name, run in runs.items():
        destination = transcript_directory / f"{name}.txt"
        shutil.copyfile(root / f"{name}.txt", destination)
        run["transcript_path"] = str(destination.relative_to(args.output.parent))
    references = {
        "q8": (root / "baseline-cache0.txt").read_text().strip(),
        "q6": (root / "q6-onepass-cache0.txt").read_text().strip(),
        "q5": (root / "q5-onepass-cache0.txt").read_text().strip(),
    }
    for name, run in runs.items():
        weight = "q6" if name.startswith("q6-") else "q5" if name.startswith("q5-") else "q8"
        candidate = (root / f"{name}.txt").read_text().strip()
        run["accuracy_vs_same_weight_one_pass"] = disagreement(references[weight], candidate)

    maximum_window = parse_run(root, "single-window163.84")
    retained_source_overhead = (
        runs["chunk122.88-context20.48-cache64"]["peak_memory_footprint_bytes"]
        - maximum_window["peak_memory_footprint_bytes"]
    )

    frames = round(6118.72 * 50)
    encoded_frames = frames // 4
    report = {
        "schema_version": 1,
        "audio_duration_seconds": 6118.72,
        "architecture_tensor_accounting": {
            "raw_float32_audio_bytes": round(6118.72 * 16000 * 4),
            "fp16_frontend_320_channels_bytes": frames * 320 * 2,
            "fp16_pre_subsampling_hidden_1024_bytes": frames * 1024 * 2,
            "fp16_pre_subsampling_expansion_4096_bytes": frames * 4096 * 2,
            "fp16_post_subsampling_hidden_1024_bytes": encoded_frames * 1024 * 2,
            "fp16_ctc_logits_16384_bytes": encoded_frames * 16384 * 2,
            "fp32_ctc_tensor_16384_bytes": encoded_frames * 16384 * 4,
            "note": "Several expansion/logit/softmax tensors coexist transiently; MLX's default cache retained freed buffers too.",
        },
        "runs": runs,
        "streaming_decode_projection": {
            "maximum_window_duration_seconds": maximum_window["audio_duration_seconds"],
            "maximum_window_peak_memory_footprint_bytes": maximum_window["peak_memory_footprint_bytes"],
            "measured_retained_full_source_overhead_bytes": retained_source_overhead,
            "status": "projected_from_isolated_maximum_window_not_full_streaming_run",
        },
        "recommended_profiles": {
            "maximum_speed": "baseline-default",
            "exact_lower_memory": "middle2048-cache0",
            "mobile_q8": "chunk122.88-context20.48-cache64",
            "mobile_q6": "q6-chunk122-context20",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
