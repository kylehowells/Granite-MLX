#!/usr/bin/env python3
"""Render PNG charts from the Granite quantization benchmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = ["fp32", "fp16", "q8", "q6", "q5", "q4", "q3", "q2"]
LABELS = ["FP32", "FP16", "Q8", "Q6", "Q5", "Q4", "Q3", "Q2"]
COLORS = ["#64748B", "#10B981", "#2563EB", "#3B82F6", "#6366F1", "#8B5CF6", "#D946EF", "#EF4444"]
BACKGROUND = "#F8FAFC"
INK = "#0F172A"
MUTED = "#64748B"
GRID = "#CBD5E1"


def style(axis, ylabel: str) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.set_ylabel(ylabel, fontsize=12, color="#334155")
    axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#94A3B8")
    axis.tick_params(axis="x", labelsize=11, colors="#334155", length=0, pad=8)
    axis.tick_params(axis="y", labelsize=10, colors=MUTED, length=0)


def title(fig, heading: str, subtitle: str) -> None:
    fig.suptitle(heading, x=0.065, y=0.985, ha="left", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.065, 0.925, subtitle, ha="left", fontsize=11.5, color=MUTED)


def label_bars(axis, bars, values, formatter, offset=0.02) -> None:
    maximum = max(values)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + maximum * offset,
            formatter(value),
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=INK,
        )


def save(fig, destination: Path) -> None:
    fig.patch.set_facecolor(BACKGROUND)
    fig.text(0.985, 0.012, "Apple M1 Max · 64 GB · full 101m 59s WAV", ha="right", fontsize=9, color=MUTED)
    fig.savefig(destination, dpi=180, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    configurations = report["configurations"]
    args.output.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(ORDER))

    sizes = [configurations[name]["model_file_bytes"] / 1e6 for name in ORDER]
    fig, axis = plt.subplots(figsize=(12, 7))
    title(fig, "Granite Speech checkpoint size", "Quantized matrix weights with FP16 residual tensors · lower is smaller")
    style(axis, "Model file (MB)")
    bars = axis.bar(x, sizes, color=COLORS, width=0.68)
    axis.set_xticks(x, LABELS)
    axis.set_ylim(0, max(sizes) * 1.18)
    label_bars(axis, bars, sizes, lambda value: f"{value:,.0f} MB")
    fig.tight_layout(rect=(0.035, 0.055, 0.99, 0.89))
    save(fig, args.output / "quantization-file-size.png")

    rss = [configurations[name]["full"]["median"]["maximum_resident_bytes"] / 1e9 for name in ORDER]
    footprint = [configurations[name]["full"]["median"]["peak_memory_footprint_bytes"] / 1e9 for name in ORDER]
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"hspace": 0.42})
    title(fig, "Granite Speech full-run memory", "Peak resident memory falls with weight size; one-pass MLX/Metal footprint is activation-dominated")
    for axis, values, ylabel, formatter in [
        (axes[0], rss, "Maximum resident memory (GB)", lambda value: f"{value:.2f}"),
        (axes[1], footprint, "Peak memory footprint (GB)", lambda value: f"{value:.1f}"),
    ]:
        style(axis, ylabel)
        bars = axis.bar(x, values, color=COLORS, width=0.68)
        axis.set_xticks(x, LABELS)
        low = 0 if axis is axes[0] else min(values) - 0.35
        axis.set_ylim(low, max(values) * (1.14 if low == 0 else 1.012))
        label_bars(axis, bars, values, formatter, offset=0.018 if low == 0 else 0.0015)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.07, top=0.88)
    save(fig, args.output / "quantization-memory.png")

    inference = [configurations[name]["full"]["median"]["inference_seconds"] for name in ORDER]
    throughput = [configurations[name]["full"]["median"]["realtime_multiple"] for name in ORDER]
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"hspace": 0.42})
    title(fig, "Granite Speech full-lecture speed", "Same input and native Swift/MLX runtime · lower inference time and higher throughput are better")
    for axis, values, ylabel, formatter in [
        (axes[0], inference, "Inference time (seconds)", lambda value: f"{value:.1f}s"),
        (axes[1], throughput, "Times realtime", lambda value: f"{value:.0f}×"),
    ]:
        style(axis, ylabel)
        bars = axis.bar(x, values, color=COLORS, width=0.68)
        axis.set_xticks(x, LABELS)
        axis.set_ylim(0, max(values) * 1.18)
        label_bars(axis, bars, values, formatter)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.07, top=0.88)
    save(fig, args.output / "quantization-speed.png")

    quantized = ["q8", "q6", "q5", "q4", "q3", "q2"]
    quant_labels = [name.upper() for name in quantized]
    quant_colors = COLORS[2:]
    disagreement = [configurations[name]["full"]["accuracy_vs_swift_fp32"]["word_disagreement_percent"] for name in quantized]
    bleu = [configurations[name]["full"]["accuracy_vs_swift_fp32"]["bleu"] for name in quantized]
    qx = np.arange(len(quantized))
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"hspace": 0.44})
    title(fig, "Granite Speech quantization accuracy", "Disagreement against native Swift FP32 · diagnostic comparison, not ground-truth WER")
    style(axes[0], "Word disagreement (%) · log scale")
    bars = axes[0].bar(qx, disagreement, color=quant_colors, width=0.68)
    axes[0].set_xticks(qx, quant_labels)
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.03, 180)
    for bar, value in zip(bars, disagreement):
        axes[0].text(bar.get_x() + bar.get_width()/2, value * 1.15, f"{value:.3g}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=INK)
    style(axes[1], "BLEU versus FP32")
    bars = axes[1].bar(qx, bleu, color=quant_colors, width=0.68)
    axes[1].set_xticks(qx, quant_labels)
    axes[1].set_ylim(0, 112)
    label_bars(axes[1], bars, bleu, lambda value: f"{value:.1f}", offset=0.012)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.07, top=0.88)
    save(fig, args.output / "quantization-accuracy.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
