#!/usr/bin/env python3
"""Run interleaved long-form benchmarks across Python, MLX, and Core ML."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def machine_state() -> dict[str, Any]:
    """Capture lightweight system state without adding benchmark load."""

    state: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "load_average": list(os.getloadavg()),
    }
    completed = subprocess.run(
        ["/usr/bin/top", "-l", "1", "-n", "0"],
        text=True,
        capture_output=True,
        check=False,
    )
    for line in completed.stdout.splitlines():
        if line.startswith(("CPU usage:", "PhysMem:", "VM:")):
            state.setdefault("top", []).append(line)
    return state


def complete(run_directory: Path, command: list[str]) -> bool:
    """Return true when the same prior command completed successfully."""

    metadata = run_directory / "run.json"
    if not metadata.is_file():
        return False
    result = json.loads(metadata.read_text())
    return result.get("return_code") == 0 and result.get("command") == command


def run_one(
    *,
    name: str,
    runtime: str,
    command: list[str],
    round_number: int,
    output: Path,
    resume: bool,
) -> None:
    """Run one timed process and retain its complete stdout and stderr."""

    run_directory = output / f"round-{round_number}" / name
    run_directory.mkdir(parents=True, exist_ok=True)
    if resume and complete(run_directory, command):
        print(f"[round {round_number}] {name}: reusing complete run", flush=True)
        return

    stdout_path = run_directory / "stdout.txt"
    stderr_path = run_directory / "stderr.txt"
    timed_command = ["/usr/bin/time", "-lp", *command]
    before = machine_state()
    print(
        f"[round {round_number}] {name}: starting; "
        f"load={before['load_average'][0]:.2f}",
        flush=True,
    )
    started = time.perf_counter()
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        completed = subprocess.run(timed_command, stdout=stdout, stderr=stderr, text=True)
    elapsed = time.perf_counter() - started
    after = machine_state()
    metadata = {
        "schema_version": 1,
        "round": round_number,
        "name": name,
        "runtime": runtime,
        "command": command,
        "return_code": completed.returncode,
        "controller_wall_seconds": elapsed,
        "machine_before": before,
        "machine_after": after,
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
    }
    (run_directory / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"[round {round_number}] {name}: finished in {elapsed:.2f}s; "
        f"load={after['load_average'][0]:.2f}; exit={completed.returncode}",
        flush=True,
    )
    if completed.returncode:
        tail = stderr_path.read_text(errors="replace")[-4000:]
        raise RuntimeError(f"{name} failed with exit {completed.returncode}:\n{tail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--mlx-fp16-model", type=Path, required=True)
    parser.add_argument("--mlx-q8-model", type=Path, required=True)
    parser.add_argument("--coreml-model", type=Path, required=True)
    parser.add_argument("--punctuation-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    required_paths = (
        args.binary,
        args.audio,
        args.source_model,
        args.mlx_fp16_model,
        args.mlx_q8_model,
        args.coreml_model,
        args.punctuation_model,
    )
    for path in required_paths:
        if not path.exists():
            parser.error(f"Missing required path: {path}")
    args.output.mkdir(parents=True, exist_ok=True)

    python_base = [
        "uv", "run", "python", "Scripts/benchmark_python_source.py",
        str(args.audio),
        "--model", str(args.source_model),
        "--source-model-id", "ibm-granite/granite-speech-5.0-470m-turboctc",
        "--source-revision", "7e74c6438b7cfb5090cb6a131538f5e8515a7de3",
        "--device", "mps", "--input-cast", "deferred",
    ]
    swift_base = [str(args.binary), "transcribe", str(args.audio)]
    raw_output = ["--no-punctuate", "--output-format", "txt", "--benchmark"]
    formatted_output = [
        "--punctuation-model", str(args.punctuation_model),
        "--output-format", "txt", "--benchmark",
    ]
    mlx_common = ["--backend", "mlx", "--mlx-cache-limit-mb", "64"]
    coreml_common = [
        "--backend", "coreml",
        "--model", str(args.source_model),
        "--coreml-model", str(args.coreml_model),
        "--coreml-compute-units", "cpu-gpu",
    ]

    # Keep this order identical in every round. Related implementations are
    # separated so no configuration receives three adjacent measurements.
    matrix: list[tuple[str, str, list[str]]] = [
        ("python-bf16-one-pass", "python-mps", [*python_base, "--precision", "bf16"]),
        (
            "swift-source-bounded", "swift-mlx",
            [*swift_base, *mlx_common, "--model", str(args.source_model), *raw_output],
        ),
        (
            "coreml-q8-bounded", "swift-coreml",
            [*swift_base, *coreml_common, *raw_output],
        ),
        ("python-fp16-one-pass", "python-mps", [*python_base, "--precision", "fp16"]),
        (
            "swift-q8-bounded", "swift-mlx",
            [*swift_base, *mlx_common, "--model", str(args.mlx_q8_model), *raw_output],
        ),
        (
            "coreml-q8-formatted", "swift-coreml",
            [*swift_base, *coreml_common, *formatted_output],
        ),
        ("python-fp32-one-pass", "python-mps", [*python_base, "--precision", "fp32"]),
        (
            "swift-fp16-one-pass", "swift-mlx",
            [
                *swift_base, *mlx_common, "--model", str(args.mlx_fp16_model),
                "--no-chunking", *raw_output,
            ],
        ),
        (
            "swift-q8-formatted", "swift-mlx",
            [
                *swift_base, *mlx_common, "--model", str(args.mlx_q8_model),
                *formatted_output,
            ],
        ),
    ]

    manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "audio": {
            "path": str(args.audio.resolve()),
            "bytes": args.audio.stat().st_size,
            "sha256": sha256(args.audio),
        },
        "rounds": args.rounds,
        "order": [name for name, _, _ in matrix],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for round_number in range(1, args.rounds + 1):
        for name, runtime, command in matrix:
            run_one(
                name=name,
                runtime=runtime,
                command=command,
                round_number=round_number,
                output=args.output,
                resume=args.resume,
            )
    print(f"Completed {args.rounds} interleaved benchmark rounds in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
