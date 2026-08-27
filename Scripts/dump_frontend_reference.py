#!/usr/bin/env python3
"""Dump the Python Granite frontend for Swift numerical-parity fixtures."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from granite_reference import load_audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("output", type=Path, help="Output .npy feature tensor")
    parser.add_argument("--model", required=True, help="Local Granite checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch

    clip = load_audio(args.audio)
    processorSource = Path(args.model) / "processing_ctc_conformer.py"
    spec = importlib.util.spec_from_file_location("granite_frontend_processor", processorSource)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {processorSource}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processor = module.CtcConformerProcessor(
        sample_rate=16_000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80,
        stack_factor=2,
        deltas=True,
        delta_win_length=3,
        logmel_floor_db=8.0,
    )
    with torch.inference_mode():
        features = processor._frontend(torch.from_numpy(clip.samples)[None, :])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, features.detach().cpu().numpy())
    print(f"wrote {args.output} shape={tuple(features.shape)} dtype={features.dtype}")


if __name__ == "__main__":
    main()
