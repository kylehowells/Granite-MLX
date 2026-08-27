#!/usr/bin/env python3
"""Aggregate punctuation and formatted ASR benchmarks into JSON and PNGs."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
from rapidfuzz.distance import Levenshtein
from sacrebleu.metrics import BLEU, CHRF


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "Benchmarks" / "punctuation"
FORMATTED = ROOT / "Benchmarks" / "formatted-comparison"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def process_wall(path: Path) -> float:
    match = re.search(r"^real\s+([0-9.]+)$", path.read_text(), re.MULTILINE)
    if not match:
        raise ValueError(f"No wall time in {path}")
    return float(match.group(1))


def normalize_lexical(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(char if char.isalnum() or char in "' " else " " for char in text)
    return " ".join(text.split())


def similarity(reference: str, candidate: str) -> dict:
    ref_words, cand_words = reference.split(), candidate.split()
    ref_lexical, cand_lexical = normalize_lexical(reference), normalize_lexical(candidate)
    ref_lex_words, cand_lex_words = ref_lexical.split(), cand_lexical.split()
    return {
        "formatted_character_similarity_percent": 100 * Levenshtein.normalized_similarity(reference, candidate),
        "formatted_word_agreement_percent": 100 * Levenshtein.normalized_similarity(ref_words, cand_words),
        "formatted_bleu": BLEU(effective_order=True).sentence_score(candidate, [reference]).score,
        "formatted_chrf2": CHRF(word_order=0).sentence_score(candidate, [reference]).score,
        "lexical_word_agreement_percent": 100 * Levenshtein.normalized_similarity(ref_lex_words, cand_lex_words),
        "reference_words": len(ref_words),
        "candidate_words": len(cand_words),
    }


def main() -> None:
    onnx = read_json(BENCH / "onnx-fp32.json")
    variants = []
    for name in ("fp16", "q8", "q6", "q5", "q4"):
        value = read_json(BENCH / f"mlx-{name}.json")
        variants.append({
            "name": name.upper(),
            "model_bytes": value["model_bytes"],
            "load_seconds": value["load_seconds"],
            "inference_seconds": value["inference_seconds"],
            "max_rss_bytes": value["max_rss_bytes"],
            "character_agreement_percent": 100 - value["character_disagreement_percent"],
            "word_agreement_percent": 100 - value["word_disagreement_percent"],
            "sentence_count": value["sentence_count"],
        })

    granite_asr_line = (FORMATTED / "granite" / "asr.time").read_text().splitlines()[0]
    granite_asr = json.loads(granite_asr_line)
    granite_formatter = read_json(FORMATTED / "granite" / "formatter.json")
    granite_text = (FORMATTED / "granite" / "formatted.txt").read_text().strip()
    parakeet_text_path = next((FORMATTED / "parakeet").glob("*.txt"))
    parakeet_text = parakeet_text_path.read_text().strip()
    current_granite_wall = process_wall(FORMATTED / "granite" / "asr.time") + process_wall(FORMATTED / "granite" / "formatter.time")
    current_parakeet_wall = process_wall(FORMATTED / "parakeet" / "process.time")

    prior_granite = read_json(ROOT / "Benchmarks" / "model-publication" / "results.json")["families"]["apache"]["checkpoints"]["q8"]["timing"]
    prior_parakeet = read_json(ROOT / "Benchmarks" / "q8-optimization" / "parakeet" / "results.json")
    representative_granite = prior_granite["total_seconds"] + granite_formatter["total_seconds"]
    representative_parakeet = prior_parakeet["model_load_seconds"] + prior_parakeet["summary"]["median_inference_seconds"]

    result = {
        "schema_version": 1,
        "audio_duration_seconds": granite_asr["audio_duration_seconds"],
        "punctuation_baseline": onnx,
        "punctuation_variants": variants,
        "formatted_asr_comparison": {
            "granite": {
                "asr_model": granite_asr["model"],
                "formatter": "MLX Q8/G64",
                "current_contended_wall_seconds": current_granite_wall,
                "representative_wall_seconds": representative_granite,
                "realtime_multiple": granite_asr["audio_duration_seconds"] / representative_granite,
                "formatted_transcript": str(FORMATTED / "granite" / "formatted.txt"),
            },
            "parakeet": {
                "model": prior_parakeet["configuration"]["model"],
                "current_contended_wall_seconds": current_parakeet_wall,
                "representative_wall_seconds": representative_parakeet,
                "realtime_multiple": granite_asr["audio_duration_seconds"] / representative_parakeet,
                "formatted_transcript": str(parakeet_text_path),
            },
            "transcript_similarity_parakeet_reference": similarity(parakeet_text, granite_text),
            "contention_note": "Current rerun coincided with sustained Backblaze CPU activity; representative values use the earlier clean ASR runs plus the newly measured Q8 formatter cost.",
        },
    }
    (FORMATTED / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    labels = [v["name"] for v in variants]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].bar(labels, [v["model_bytes"] / 1_000_000 for v in variants], color="#5B8FF9")
    axes[0, 0].set_title("Formatter checkpoint size")
    axes[0, 0].set_ylabel("MB")
    axes[0, 1].bar(labels, [v["inference_seconds"] for v in variants], color="#61DDAA")
    axes[0, 1].set_title("101m59s transcript formatting time")
    axes[0, 1].set_ylabel("Seconds (model inference)")
    axes[1, 0].bar(labels, [v["max_rss_bytes"] / 1_000_000 for v in variants], color="#65789B")
    axes[1, 0].set_title("Peak process RSS")
    axes[1, 0].set_ylabel("MB")
    accuracy = [v["character_agreement_percent"] for v in variants]
    axes[1, 1].bar(labels, accuracy, color="#F6BD16")
    axes[1, 1].set_ylim(min(accuracy) - 0.1, 100.01)
    axes[1, 1].set_title("Character agreement with ONNX FP32")
    axes[1, 1].set_ylabel("Percent")
    figure.suptitle("MLX punctuation/capitalization conversion", fontsize=16)
    figure.savefig(BENCH / "mlx-quantization-results.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    times = [representative_granite, representative_parakeet]
    bars = axis.bar(["Granite Q8 + MLX Q8 cleanup", "Parakeet MLX v2"], times, color=["#5B8FF9", "#61DDAA"])
    axis.set_ylabel("Seconds, complete 101m59s recording")
    axis.set_title("Formatted transcription: representative end-to-end time")
    for bar, value in zip(bars, times):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}s", ha="center")
    figure.savefig(FORMATTED / "formatted-speed-comparison.png", dpi=180)
    plt.close(figure)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
