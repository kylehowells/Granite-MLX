#!/usr/bin/env python3
"""Render the checked three-round backend benchmark chart."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "Benchmarks" / "coreml"
MATRIX = ROOT / "Benchmarks" / "backend-matrix"


CONFIGURATIONS = [
    ("python-bf16-one-pass", "Python BF16\none pass", "#787878"),
    ("python-fp16-one-pass", "Python FP16\none pass", "#9a9a9a"),
    ("python-fp32-one-pass", "Python FP32\none pass", "#b4b4b4"),
    ("swift-source-bounded", "Swift source\nbounded", "#7656a8"),
    ("swift-fp16-one-pass", "Swift FP16\none pass", "#9a76c4"),
    ("swift-q8-bounded", "MLX Q8\nbounded", "#2878b5"),
    ("coreml-q8-bounded", "Core ML Q8\nbounded", "#269f8f"),
]


def values(result: dict, metric: str, divisor: float = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return medians and asymmetric min/max errors for one metric."""

    summaries = [
        result["configurations"][key]["summary"][metric]
        for key, _, _ in CONFIGURATIONS
    ]
    medians = np.asarray([summary["median"] / divisor for summary in summaries])
    minimums = np.asarray([summary["minimum"] / divisor for summary in summaries])
    maximums = np.asarray([summary["maximum"] / divisor for summary in summaries])
    return medians, np.vstack((medians - minimums, maximums - medians))


def main() -> None:
    result = json.loads((MATRIX / "results.json").read_text())
    labels = [label for _, label, _ in CONFIGURATIONS]
    colors = [color for _, _, color in CONFIGURATIONS]
    positions = np.arange(len(labels))

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 7.2), sharey=True)
    panels = [
        ("speech_inference_seconds", 1, "Speech inference", "Seconds"),
        ("process_wall_seconds", 1, "Complete process wall", "Seconds"),
        ("peak_physical_footprint_bytes", 1e9, "Peak memory footprint", "GB"),
    ]
    for axis, (metric, divisor, title, unit) in zip(axes, panels):
        medians, errors = values(result, metric, divisor)
        bars = axis.barh(
            positions,
            medians,
            xerr=errors,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
            error_kw={"ecolor": "#333333", "capsize": 3, "elinewidth": 1},
        )
        axis.set_title(f"{title}\n(lower is better)", fontsize=12, weight="semibold")
        axis.set_xlabel(unit)
        axis.bar_label(
            bars,
            labels=[f"{value:.2f}" for value in medians],
            padding=4,
            fontsize=9,
        )
        axis.set_xlim(0, max(medians + errors[1]) * 1.22)
        axis.grid(axis="y", visible=False)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()

    figure.suptitle(
        "Granite Speech 5.0 — M1 Max backend comparison",
        fontsize=16,
        weight="semibold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.015,
        "6,118.72-second lecture · median of 3 interleaved clean processes · whiskers show min–max",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.94))
    figure.savefig(BENCHMARK / "coreml-vs-mlx.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
