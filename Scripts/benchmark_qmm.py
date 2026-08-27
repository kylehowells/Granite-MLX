#!/usr/bin/env python3
"""Microbenchmark Granite's Linear shapes across MLX quantized paths."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx


SHAPES = (
    ("input", 320, 1024),
    ("hidden", 1024, 1024),
    ("ffn_up", 1024, 4096),
    ("ffn_down", 4096, 1024),
    ("conv_glu", 1024, 4096),
    ("ctc", 1024, 16384),
    ("ctc_feedback", 16384, 1024),
)


SYMMETRIC_Q8_KERNEL = mx.fast.metal_kernel(
    name="granite_symmetric_q8_g128",
    input_names=["x", "w", "scales"],
    output_names=["out"],
    source="""
        uint lane = thread_position_in_threadgroup.x;
        uint output_column = threadgroup_position_in_grid.x;
        uint output_row = threadgroup_position_in_grid.y * 256 + lane;
        threadgroup half weight_tile[128];
        threadgroup half group_scale;
        float sum = 0.0f;
        for (uint k0 = 0; k0 < K; k0 += 128) {
            if (lane < 128) {
                weight_tile[lane] = half(w[output_column * K + k0 + lane]);
            }
            if (lane == 0) {
                group_scale = scales[output_column * (K / 128) + k0 / 128];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (output_row < M) {
                for (uint j = 0; j < 128; ++j) {
                    sum += float(x[output_row * K + k0 + j])
                        * float(weight_tile[j]) * float(group_scale);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (output_row < M) {
            out[output_row * N + output_column] = half(sum);
        }
    """,
)


def timed(operation: Callable[[], mx.array], repetitions: int) -> list[float]:
    mx.eval(operation())
    values = []
    for _ in range(repetitions):
        start = time.perf_counter()
        mx.eval(operation())
        values.append(time.perf_counter() - start)
    return values


def summary(values: list[float]) -> dict[str, Any]:
    median = statistics.median(values)
    return {
        "runs_seconds": values,
        "median_seconds": median,
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def capability_probe() -> dict[str, Any]:
    x = mx.ones((4, 128), dtype=mx.float16)
    w = mx.ones((256, 128), dtype=mx.float16)
    results: dict[str, Any] = {}

    try:
        a = mx.ones((4, 32), dtype=mx.int8)
        b = mx.ones((8, 32), dtype=mx.int8)
        mx.eval(a @ b.T)
        results["int8_matmul"] = {"supported": True}
    except Exception as error:
        results["int8_matmul"] = {"supported": False, "error": str(error)}

    for mode, group_size in (("affine", 64), ("mxfp8", 32)):
        key = f"qqmm_{mode}"
        try:
            packed, scales, *rest = mx.quantize(
                w, group_size=group_size, bits=8, mode=mode
            )
            output = mx.qqmm(
                x, packed, scales,
                group_size=group_size, bits=8, mode=mode,
            )
            mx.eval(output)
            results[key] = {"supported": True, "output_dtype": str(output.dtype)}
        except Exception as error:
            results[key] = {"supported": False, "error": str(error)}
    return results


def benchmark_shape(
    name: str, input_size: int, output_size: int,
    frame_counts: list[int], repetitions: int,
) -> dict[str, Any]:
    # Deterministic values make numerical comparisons reproducible while still
    # resembling normally distributed trained parameters.
    mx.random.seed(17 + input_size + output_size)
    weight = mx.random.normal((output_size, input_size)).astype(mx.float16) * 0.02
    mx.eval(weight)

    variants: dict[str, tuple[Callable[[mx.array], mx.array], mx.array]] = {
        "fp16": (lambda x: x @ weight.T, weight),
    }
    for group_size in (64, 128):
        if input_size % group_size:
            continue
        packed, scales, biases = mx.quantize(
            weight, group_size=group_size, bits=8, mode="affine"
        )
        mx.eval(packed, scales, biases)
        variants[f"q8_affine_g{group_size}"] = (
            lambda x, p=packed, s=scales, b=biases, g=group_size: mx.quantized_matmul(
                x, p, s, b, transpose=True, group_size=g, bits=8, mode="affine"
            ),
            packed,
        )

    if input_size % 128 == 0:
        grouped = weight.reshape(output_size, input_size // 128, 128)
        symmetric_scales = mx.max(mx.abs(grouped), axis=-1) / 127
        symmetric_weights = mx.clip(
            mx.round(grouped / symmetric_scales[..., None]), -127, 127
        ).astype(mx.int8).reshape(output_size, input_size)
        symmetric_scales = symmetric_scales.astype(mx.float16)
        mx.eval(symmetric_weights, symmetric_scales)

        def symmetric_q8(x: mx.array) -> mx.array:
            frames = x.size // input_size
            return SYMMETRIC_Q8_KERNEL(
                inputs=[x, symmetric_weights, symmetric_scales],
                template=[("M", frames), ("N", output_size), ("K", input_size)],
                grid=(output_size * 256, (frames + 255) // 256, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(1, frames, output_size)],
                output_dtypes=[mx.float16],
            )[0]

        variants["q8_symmetric_g128_custom"] = (symmetric_q8, symmetric_weights)

    # MXFP8 weight-only QMM is measurable on older Apple GPUs even when the
    # activation-quantized QQMM path is unavailable. It is labelled separately.
    if input_size % 32 == 0:
        packed, scales = mx.quantize(
            weight, group_size=32, bits=8, mode="mxfp8"
        )
        mx.eval(packed, scales)
        variants["mxfp8_weight_only_g32"] = (
            lambda x, p=packed, s=scales: mx.quantized_matmul(
                x, p, s, None, transpose=True,
                group_size=32, bits=8, mode="mxfp8",
            ),
            packed,
        )

    cases: dict[str, Any] = {}
    for frames in frame_counts:
        x = mx.random.normal((1, frames, input_size)).astype(mx.float16)
        mx.eval(x)
        reference = variants["fp16"][0](x)
        mx.eval(reference)
        frame_result: dict[str, Any] = {}
        for variant_name, (operation, _) in variants.items():
            try:
                output = operation(x)
                mx.eval(output)
                difference = mx.abs(output.astype(mx.float32) - reference.astype(mx.float32))
                max_error = mx.max(difference).item()
                mean_error = mx.mean(difference).item()
                timings = timed(lambda op=operation, value=x: op(value), repetitions)
                frame_result[variant_name] = {
                    **summary(timings),
                    "max_absolute_error_vs_fp16": max_error,
                    "mean_absolute_error_vs_fp16": mean_error,
                    "output_dtype": str(output.dtype),
                }
            except Exception as error:
                frame_result[variant_name] = {"error": str(error)}
            mx.clear_cache()
        cases[str(frames)] = frame_result
    return {
        "name": name,
        "input_size": input_size,
        "output_size": output_size,
        "frames": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[32, 128, 512, 2048])
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    device = mx.device_info()
    report = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "mlx_version": mx.__version__,
            "platform": platform.platform(),
            "device": device,
        },
        "methodology": {
            "repetitions": args.repetitions,
            "frames": args.frames,
            "timing": "Warm once, synchronize each operation with mx.eval, report fresh-output wall time.",
            "accuracy": "Absolute output error against the same random FP16 matrix multiplication.",
        },
        "capabilities": capability_probe(),
        "shapes": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for shape in SHAPES:
        print(f"Benchmarking {shape[0]} {shape[1]}->{shape[2]}", flush=True)
        report["shapes"].append(benchmark_shape(*shape, args.frames, args.repetitions))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
