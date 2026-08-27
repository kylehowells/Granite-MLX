#!/usr/bin/env python3
"""Render PNG charts from the consolidated Q8 optimization JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = ["#64748b", "#2563eb", "#7c3aed", "#dc2626", "#0891b2"]


def bar_chart(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    bars = ax.bar(labels, values, color=COLORS[:len(labels)])
    ax.set_title(title, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    paired = report["paired_full_input_results"]
    experiments = report["experimental_full_input_results"]

    paired_labels = ["Q8 G64\nFP16", "Q8 G128\nFP16"]
    paired_entries = [paired["q8_g64_fp16"], paired["q8_g128_fp16"]]
    bar_chart(
        args.output / "full-inference-speed.png",
        "Granite Q8 — paired 101m59s inference on M1 Max",
        paired_labels,
        [entry["median"]["inference_seconds"] for entry in paired_entries],
        "Seconds (lower is better)",
    )
    bar_chart(
        args.output / "peak-memory-footprint.png",
        "Granite Q8 — paired macOS peak memory footprint",
        paired_labels,
        [entry["median"]["peak_memory_footprint_bytes"] / 1e9 for entry in paired_entries],
        "GB (decimal)",
    )
    experimental_labels = ["FP8 emulated", "INT8 emulated", "CTC tiled"]
    experimental_entries = [
        experiments["q8_g128_fp8_emulated"],
        experiments["q8_g128_int8_emulated"],
        experiments["q8_g128_fp16_ctc_tile_2048"],
    ]
    bar_chart(
        args.output / "experimental-speed-overhead.png",
        "Experimental modes — time versus adjacent FP16 control",
        experimental_labels,
        [(entry["inference_time_ratio_vs_paired_control"] - 1) * 100 for entry in experimental_entries],
        "Inference time overhead (%)",
    )
    accuracy_values = [
        paired["q8_g64_fp16"]["accuracy_vs_swift_fp32_weights"]["word_disagreement_percent"],
        paired["q8_g128_fp16"]["accuracy_vs_swift_fp32_weights"]["word_disagreement_percent"],
        experiments["q8_g128_fp8_emulated"]["accuracy_vs_q8_g128_fp16"]["word_disagreement_percent"],
        experiments["q8_g128_int8_emulated"]["accuracy_vs_q8_g128_fp16"]["word_disagreement_percent"],
        experiments["q8_g128_fp16_ctc_tile_2048"]["accuracy_vs_q8_g128_fp16"]["word_disagreement_percent"],
    ]
    bar_chart(
        args.output / "transcript-disagreement.png",
        "Granite Q8 — transcript disagreement (diagnostic, not WER)",
        ["Q8 G64\nvs FP32", "Q8 G128\nvs FP32", *experimental_labels],
        accuracy_values, "Word disagreement (%)",
    )
    files = report["model_files"]
    bar_chart(
        args.output / "model-file-size.png",
        "Granite Q8 checkpoint size",
        ["Group 64", "Mixed Group 128"],
        [files["q8_g64_bytes"] / 2**20, files["q8_g128_mixed_bytes"] / 2**20],
        "MiB",
    )
    parakeet_path = args.results.parent / "parakeet/results.json"
    if parakeet_path.exists():
        parakeet = json.loads(parakeet_path.read_text())
        granite = paired["q8_g128_fp16"]["median"]
        parakeet_summary = parakeet["summary"]
        comparison_labels = ["Granite Q8 G128\none pass", "Parakeet TDT 0.6B v2\n120s chunks"]
        bar_chart(
            args.output / "granite-vs-parakeet-speed.png",
            "101m59s lecture — Granite-MLX versus Parakeet-MLX",
            comparison_labels,
            [granite["inference_seconds"], parakeet_summary["median_inference_seconds"]],
            "Inference seconds (lower is better)",
        )
        bar_chart(
            args.output / "granite-vs-parakeet-throughput.png",
            "101m59s lecture — real-time throughput on M1 Max",
            comparison_labels,
            [granite["realtime_multiple"], parakeet_summary["median_realtime_multiple"]],
            "Audio seconds per wall second (higher is better)",
        )
    print(f"Wrote PNG charts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
