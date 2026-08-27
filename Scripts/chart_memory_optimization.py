#!/usr/bin/env python3
"""Render Granite memory-optimization PNG charts from results JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = ["#64748b", "#2563eb", "#7c3aed", "#0891b2", "#16a34a", "#ea580c"]


def bars(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
    objects = ax.bar(labels, values, color=COLORS[:len(labels)])
    ax.set_title(title, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(objects, fmt="%.2f", padding=3)
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
    runs = report["runs"]
    args.output.mkdir(parents=True, exist_ok=True)

    names = [
        "baseline-default", "baseline-cache0", "middle2048-cache0",
        "chunk122.88", "chunk122.88-context20.48-cache64",
    ]
    labels = ["One pass\ndefault cache", "One pass\ncache off", "One pass\nmiddle tiled", "122.88s\nno context", "122.88s + 20.48s\nmobile profile"]
    memory_labels = [*labels, "Projected with\nstreamed source"]
    memory_values = [runs[name]["peak_memory_footprint_bytes"] / 1e9 for name in names]
    memory_values.append(
        report["streaming_decode_projection"]["maximum_window_peak_memory_footprint_bytes"] / 1e9
    )
    bars(
        args.output / "memory-footprint-reduction.png",
        "Granite Q8 — 101m59s peak memory reduction on M1 Max",
        memory_labels,
        memory_values,
        "macOS peak footprint (GB, lower is better)",
    )
    bars(
        args.output / "memory-speed-tradeoff.png",
        "Granite Q8 — inference time after memory optimization",
        labels,
        [runs[name]["inference_seconds"] for name in names],
        "Inference seconds (lower is better)",
    )
    tensor = report["architecture_tensor_accounting"]
    bars(
        args.output / "large-tensor-sizes.png",
        "Granite one-pass — individual long-input tensor sizes",
        ["Raw audio\nFP32", "Frontend\nFP16", "Hidden before\nsubsampling", "4096-channel\nexpansion", "CTC logits\nFP16", "CTC softmax\nFP32"],
        [
            tensor["raw_float32_audio_bytes"] / 1e9,
            tensor["fp16_frontend_320_channels_bytes"] / 1e9,
            tensor["fp16_pre_subsampling_hidden_1024_bytes"] / 1e9,
            tensor["fp16_pre_subsampling_expansion_4096_bytes"] / 1e9,
            tensor["fp16_ctc_logits_16384_bytes"] / 1e9,
            tensor["fp32_ctc_tensor_16384_bytes"] / 1e9,
        ],
        "Tensor size (GB, decimal)",
    )
    bars(
        args.output / "mobile-weight-options.png",
        "Bounded 122.88s + 20.48s context profile",
        ["Q8", "Q6", "Q5"],
        [
            runs["chunk122.88-context20.48-cache64"]["peak_memory_footprint_bytes"] / 1e9,
            runs["q6-chunk122-context20"]["peak_memory_footprint_bytes"] / 1e9,
            runs["q5-chunk122-context20"]["peak_memory_footprint_bytes"] / 1e9,
        ],
        "macOS peak footprint (GB)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
