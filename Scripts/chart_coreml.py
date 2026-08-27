#!/usr/bin/env python3
"""Render the checked Core ML versus MLX benchmark chart."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "Benchmarks" / "coreml"


def main() -> None:
    result = json.loads((BENCHMARK / "results.json").read_text())
    mlx = result["mlx_release_baseline"]
    coreml = result["winning_configuration"]
    formatted = result["formatted_default_run"]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    colors = ["#8c8c8c", "#2878b5"]

    names = ["MLX Q8", "Core ML Q8"]
    values = [mlx["inference_seconds"], coreml["inference_seconds"]]
    bars = axes[0].bar(names, values, color=colors)
    axes[0].set_title("Speech inference\n(lower is better)")
    axes[0].set_ylabel("Seconds for 101m59s audio")
    axes[0].bar_label(bars, fmt="%.2f s", padding=3)
    axes[0].set_ylim(0, max(values) * 1.22)

    memory = [
        mlx["peak_memory_footprint_bytes"] / 1e9,
        coreml["peak_memory_footprint_bytes"] / 1e9,
    ]
    bars = axes[1].bar(names, memory, color=colors)
    axes[1].set_title("Peak memory footprint\n(lower is better)")
    axes[1].set_ylabel("GB")
    axes[1].bar_label(bars, fmt="%.2f GB", padding=3)
    axes[1].set_ylim(0, max(memory) * 1.25)

    agreement = [
        100 - coreml["accuracy_vs_mlx_default"]["word_disagreement_percent"],
        formatted["lexical_word_agreement_vs_mlx_percent"],
    ]
    agreement_names = ["Raw words", "Formatted lexical"]
    bars = axes[2].bar(agreement_names, agreement, color=["#2878b5", "#3a9d5d"])
    axes[2].set_title("Core ML agreement with MLX\n(higher is better)")
    axes[2].set_ylabel("Agreement (%)")
    axes[2].set_ylim(99, 100)
    axes[2].bar_label(bars, fmt="%.3f%%", padding=3)

    figure.suptitle("Granite Speech 5.0 — M1 Max Core ML optimization", fontsize=15)
    figure.tight_layout()
    figure.savefig(BENCHMARK / "coreml-vs-mlx.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
