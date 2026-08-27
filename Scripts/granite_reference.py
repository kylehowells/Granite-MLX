#!/usr/bin/env python3
"""Reference CLI for Granite Speech 5.0 TurboCTC.

This intentionally defines the command-line contract that the native Swift
implementation must match. It uses the original Transformers/PyTorch model,
and is not part of the eventual runtime dependency set.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import re
import resource
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_MODEL = "mlx-community/granite-speech-5.0-470m-turboctc"
SAMPLE_RATE = 16_000
FRAME_RATE = 12.5  # 100 Hz frontend / 8x temporal subsampling


@dataclass
class AudioClip:
    path: Path
    samples: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def load_audio(path: Path, target_sample_rate: int = SAMPLE_RATE) -> AudioClip:
    import numpy as np
    import soundfile as sf

    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = np.asarray(samples, dtype=np.float32)

    if sample_rate != target_sample_rate:
        import torch
        import torchaudio.functional as F

        tensor = torch.from_numpy(samples)
        samples = F.resample(tensor, sample_rate, target_sample_rate).numpy()
        sample_rate = target_sample_rate

    return AudioClip(path=path, samples=samples, sample_rate=sample_rate)


def memory_report(device: str) -> dict[str, Any]:
    """Collect process and MPS allocator memory after transcription.

    macOS process footprint and MPS allocator values describe different memory
    accounting views, so they are reported separately rather than summed.
    """

    report: dict[str, Any] = {
        "maximum_resident_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if device == "mps":
        import torch

        torch.mps.synchronize()
        report.update(
            {
                "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
                "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
                "mps_recommended_max_bytes": torch.mps.recommended_max_memory(),
            }
        )

    if sys.platform == "darwin" and Path("/usr/bin/footprint").is_file():
        completed = subprocess.run(
            [
                "/usr/bin/footprint",
                "-p",
                str(os.getpid()),
                "-f",
                "bytes",
                "--noCategories",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for key, output_key in (
            ("phys_footprint", "physical_footprint_bytes"),
            ("phys_footprint_peak", "peak_physical_footprint_bytes"),
        ):
            match = re.search(rf"^\s*{key}:\s+(\d+) B$", completed.stdout, re.MULTILINE)
            if match:
                report[output_key] = int(match.group(1))
    return report


def load_reference_model(model_id: str, device: str, dtype: str):
    """Load the current local/reference Transformers implementation.

    Granite 5.0's repository has used custom model code while the upstream
    Transformers integration is being released. Local model directories from
    the development checkout therefore use the bundled CtcConformer classes.
    For a published compatible repository, AutoModelForCTC is attempted first.
    """

    import torch

    model_path = Path(model_id).expanduser()
    if model_path.is_dir() and (model_path / "modeling_ctc_conformer.py").exists():
        # Current Transformers source includes the native Granite 5.0 classes.
        # Prefer those; the fallback below keeps this reference usable with
        # earlier development snapshots that shipped a local wrapper class.
        try:
            from transformers import AutoModelForCTC, AutoProcessor

            processor = AutoProcessor.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=True
            )
            model = AutoModelForCTC.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=True
            )
        except (ImportError, KeyError, ValueError, RuntimeError):
            processor = None
            model = None

    else:
        processor = None
        model = None

    if model is None or processor is None:
        # The downloaded development model uses relative imports, but its
        # directory name is not a valid Python package name. Load the bundled
        # files under a synthetic package so those imports still work.
        package_name = "_granite_reference_model"
        package = types.ModuleType(package_name)
        package.__path__ = [str(model_path)]
        sys.modules[package_name] = package

        def load_module(module_name: str):
            qualified_name = f"{package_name}.{module_name}"
            spec = importlib.util.spec_from_file_location(
                qualified_name, model_path / f"{module_name}.py"
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load {module_name}.py from {model_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified_name] = module
            spec.loader.exec_module(module)
            return module

        configuration_module = load_module("configuration_ctc_conformer")
        modeling_module = load_module("modeling_ctc_conformer")
        processing_module = load_module("processing_ctc_conformer")
        CtcConformerConfig = configuration_module.CtcConformerConfig
        CtcConformerForCTC = modeling_module.CtcConformerForCTC
        CtcConformerProcessor = processing_module.CtcConformerProcessor

        config = CtcConformerConfig.from_pretrained(model_path)
        model = CtcConformerForCTC.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
        )
        processor = CtcConformerProcessor(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=80,
            stack_factor=2,
            deltas=True,
            delta_win_length=3,
            logmel_floor_db=8.0,
            tokenizer_path=str(model_path / "tokenizer.json"),
        )
    if model is None or processor is None:
        from transformers import AutoModelForCTC, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        model = AutoModelForCTC.from_pretrained(
            model_id, trust_remote_code=True
        )

    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()
    return model, processor


def ctc_tokens(logits: Any) -> list[int]:
    """Collapse greedy CTC IDs, retaining token IDs in emission order."""

    import torch

    ids = logits.argmax(dim=-1).reshape(-1).tolist()
    result: list[int] = []
    previous = -1
    for token_id in ids:
        if token_id != previous and token_id != 0:
            result.append(int(token_id))
        previous = token_id
    return result


def ctc_token_frames(logits: Any, processor: Any, offset: float = 0.0) -> list[dict[str, Any]]:
    """Return collapsed CTC tokens with approximate audio-frame timings."""

    ids = logits.argmax(dim=-1).reshape(-1).tolist()
    result: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None

    def finish(token: dict[str, Any] | None) -> None:
        if token is None:
            return
        start = offset + token["frame"] / FRAME_RATE
        end = offset + (token["end_frame"] + 1) / FRAME_RATE
        token["start"] = round(start, 6)
        token["end"] = round(end, 6)
        token["text"] = decode_token_piece(processor, token["id"])
        result.append(token)

    previous = 0
    for frame, token_id in enumerate(ids):
        token_id = int(token_id)
        if token_id == 0:
            finish(active)
            active = None
        elif token_id != previous:
            finish(active)
            active = {"id": token_id, "frame": frame, "end_frame": frame}
        elif active is not None:
            active["end_frame"] = frame
        previous = token_id
    finish(active)
    return result


def decode_token_piece(processor: Any, token_id: int) -> str:
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode([token_id], skip_special_tokens=True)
    return processor.decode([token_id], skip_special_tokens=True)


def tokens_to_words(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group tokenizer pieces into approximate word alignments."""

    words: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for token in tokens:
        piece = token.get("text", "")
        if not piece:
            continue
        # Granite's tokenizer uses whitespace-bearing pieces. Handling split
        # pieces as well keeps this usable if the tokenizer vocabulary changes.
        parts = re.split(r"(\s+)", piece)
        for part in parts:
            if not part:
                continue
            if part.isspace():
                if current is not None:
                    words.append(current)
                    current = None
                continue
            if current is None:
                current = {"text": part, "start": token["start"], "end": token["end"]}
            else:
                current["text"] += part
                current["end"] = token["end"]
    if current is not None:
        words.append(current)

    # A CTC token's emission frame is a reliable onset estimate, but its
    # final active frame often ends well before the spoken word ends. For
    # subtitle highlighting, use the next word onset as the end boundary when
    # the gap is plausibly part of the same utterance. This mirrors the
    # convention used by Parakeet MLX's highlighted SRT/VTT exporter.
    for index in range(len(words) - 1):
        next_start = words[index + 1]["start"]
        if next_start - words[index]["start"] <= 2.0:
            words[index]["end"] = next_start
    return words


def merge_aligned_words(existing: list[dict[str, Any]], addition: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge overlapping chunk words while preserving their timestamps."""

    if not existing:
        return addition
    if not addition:
        return existing
    left = [w["text"].lower() for w in existing]
    right = [w["text"].lower() for w in addition]
    for size in range(min(12, len(left), len(right)), 0, -1):
        if left[-size:] == right[:size]:
            return existing + addition[size:]
    return existing + addition


def segment_words(
    words: list[dict[str, Any]],
    max_words: int = 20,
    silence_gap: float = 0.8,
    max_duration: float = 8.0,
) -> list[dict[str, Any]]:
    """Create readable subtitle cues from word alignments."""

    if not words:
        return []
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for word in words:
        split = bool(current) and (
            word["start"] - current[-1]["start"] >= silence_gap
            or len(current) >= max_words
            or word["start"] - current[0]["start"] >= max_duration
        )
        current.append(word)
        if split:
            finished = current[:-1]
            segments.append({"text": " ".join(w["text"] for w in finished),
                             "start": finished[0]["start"], "end": finished[-1]["end"],
                             "words": finished})
            current = [word]
    if current:
        segments.append({"text": " ".join(w["text"] for w in current),
                         "start": current[0]["start"], "end": current[-1]["end"],
                         "words": current})
    return segments


def decode_tokens(processor: Any, token_ids: list[int]) -> str:
    if hasattr(processor, "decode"):
        return processor.decode(token_ids).strip()
    return processor.batch_decode([token_ids], skip_special_tokens=True)[0].strip()


def transcribe_clip(model: Any, processor: Any, clip: AudioClip, device: str) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    waveform = [clip.samples]
    inputs = processor(waveform, sampling_rate=clip.sample_rate, device=device)
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)

    with torch.inference_mode():
        if hasattr(model, "transcribe"):
            output = model.transcribe(**inputs)
            logits = output.logits
            token_ids = output.preds[0]
        else:
            # GraniteSpeech5ForCTC.generate() returns already-decoded token
            # IDs. Use the forward pass here so the reference retains logits
            # for parity checks and emission-frame diagnostics.
            output = model(**inputs)
            logits = output.logits
            token_ids = ctc_tokens(logits)

    elapsed = time.perf_counter() - started
    text = decode_tokens(processor, token_ids)
    tokens = ctc_token_frames(logits, processor)
    words = tokens_to_words(tokens)

    return {
        "audio_file": str(clip.path),
        "text": text,
        "duration_seconds": round(clip.duration, 6),
        "inference_seconds": round(elapsed, 6),
        "real_time_factor": round(elapsed / clip.duration, 6) if clip.duration else None,
        "token_count": len(token_ids),
        "tokens": tokens,
        "words": words,
        "segments": segment_words(words),
        "timing_note": "Word timings are approximate CTC emission-frame alignments.",
    }


def merge_chunk_text(existing: str, addition: str) -> str:
    """Join chunk transcripts while removing a small repeated word overlap."""

    existing_words = existing.split()
    addition_words = addition.split()
    if not existing_words:
        return addition.strip()
    if not addition_words:
        return existing.strip()
    for size in range(min(12, len(existing_words), len(addition_words)), 0, -1):
        if existing_words[-size:] == addition_words[:size]:
            addition_words = addition_words[size:]
            break
    return " ".join(existing_words + addition_words).strip()


def transcribe_long_clip(
    model: Any,
    processor: Any,
    clip: AudioClip,
    device: str,
    chunk_duration: float,
    overlap_duration: float,
) -> dict[str, Any]:
    chunk_samples = max(1, round(chunk_duration * clip.sample_rate))
    overlap_samples = max(0, round(overlap_duration * clip.sample_rate))
    step = chunk_samples - overlap_samples
    chunks: list[dict[str, Any]] = []
    tokens: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    text = ""
    total_inference = 0.0
    start = 0
    while start < len(clip.samples):
        end = min(start + chunk_samples, len(clip.samples))
        chunk = AudioClip(
            path=clip.path,
            samples=clip.samples[start:end],
            sample_rate=clip.sample_rate,
        )
        result = transcribe_clip(model, processor, chunk, device)
        chunk_text = result["text"]
        text = merge_chunk_text(text, chunk_text)
        total_inference += result["inference_seconds"]
        offset = start / clip.sample_rate
        chunk_tokens = [
            {**token, "start": round(token["start"] + offset, 6),
             "end": round(token["end"] + offset, 6)}
            for token in result["tokens"]
        ]
        chunk_words = [
            {**word, "start": round(word["start"] + offset, 6),
             "end": round(word["end"] + offset, 6)}
            for word in result["words"]
        ]
        tokens.extend(chunk_tokens)
        words = merge_aligned_words(words, chunk_words)
        chunks.append(
            {
                "start": round(offset, 6),
                "end": round(end / clip.sample_rate, 6),
                "text": chunk_text,
            }
        )
        if end == len(clip.samples):
            break
        start += step

    return {
        "audio_file": str(clip.path),
        "text": text,
        "duration_seconds": round(clip.duration, 6),
        "inference_seconds": round(total_inference, 6),
        "real_time_factor": round(total_inference / clip.duration, 6) if clip.duration else None,
        "token_count": len(tokens),
        "tokens": tokens,
        "words": words,
        "segments": segment_words(words),
        "chunks": chunks,
        "timing_note": "Word timings are approximate CTC emission-frame alignments with chunk offsets.",
    }


def format_timestamp(seconds: float, decimal: str = ",") -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{decimal}{milliseconds:03d}"


def cue_text(words: list[dict[str, Any]], highlight_index: int | None = None, tag: str = "") -> str:
    values = []
    for index, word in enumerate(words):
        value = word["text"]
        if index == highlight_index:
            value = f"{tag}{value}{'</u>' if tag == '<u>' else '</b>'}"
        values.append(value)
    return " ".join(values)


def format_srt(result: dict[str, Any], highlight_words: bool = False) -> str:
    segments = result.get("segments", [])
    if not segments:
        text = result["text"] or "[no speech detected]"
        end = max(0.001, result["duration_seconds"])
        return f"1\n{format_timestamp(0)} --> {format_timestamp(end)}\n{text}\n"
    output: list[str] = []
    index = 1
    for segment in segments:
        words = segment["words"]
        highlights = range(len(words)) if highlight_words else [None]
        for highlight in highlights:
            start = words[highlight]["start"] if highlight is not None else segment["start"]
            end = words[highlight]["end"] if highlight is not None else segment["end"]
            output.extend([str(index), f"{format_timestamp(start)} --> {format_timestamp(end)}"])
            output.extend([cue_text(words, highlight, "<u>" if highlight is not None else ""), ""])
            index += 1
    return "\n".join(output)


def format_vtt(result: dict[str, Any], highlight_words: bool = False) -> str:
    segments = result.get("segments", [])
    if not segments:
        text = result["text"] or "[no speech detected]"
        end = max(0.001, result["duration_seconds"])
        return f"WEBVTT\n\n00:00:00.000 --> {format_timestamp(end, decimal='.')}\n{text}\n"
    output = ["WEBVTT", ""]
    for segment in segments:
        words = segment["words"]
        highlights = range(len(words)) if highlight_words else [None]
        for highlight in highlights:
            start = words[highlight]["start"] if highlight is not None else segment["start"]
            end = words[highlight]["end"] if highlight is not None else segment["end"]
            output.append(f"{format_timestamp(start, decimal='.')} --> {format_timestamp(end, decimal='.')}" )
            output.extend([cue_text(words, highlight, "<b>" if highlight is not None else ""), ""])
    return "\n".join(output)


def output_result(result: dict[str, Any], output_format: str, highlight_words: bool = False) -> str:
    if output_format == "txt":
        return result["text"] + "\n"
    if output_format == "srt":
        return format_srt(result, highlight_words)
    if output_format == "vtt":
        return format_vtt(result, highlight_words)
    if output_format == "json":
        return json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    raise ValueError(f"Unsupported output format: {output_format}")


def output_path(result: dict[str, Any], output_dir: Path, template: str, fmt: str, index: int) -> Path:
    source = Path(result["audio_file"])
    stem = source.stem
    filename = template.format(
        filename=stem,
        parent=str(source.parent),
        index=index,
        date=datetime.datetime.now().strftime("%Y%m%d"),
    )
    return output_dir / f"{filename}.{fmt}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="granite-reference",
        description="Reference Granite Speech 5.0 TurboCTC transcription CLI.",
    )
    parser.add_argument("audio", nargs="+", type=Path, help="Audio file(s) to transcribe.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local model directory.")
    parser.add_argument("--output-dir", type=Path, help="Directory for output files; stdout is used when omitted.")
    parser.add_argument("--output-format", choices=["txt", "srt", "vtt", "json", "all"], default="srt")
    parser.add_argument("--output-template", default="{filename}", help="Output filename template without extension.")
    parser.add_argument("--chunk-duration", type=float, default=0.0, help="Chunk length in seconds; 0 disables chunking.")
    parser.add_argument("--overlap-duration", type=float, default=15.0, help="Overlap between long-audio chunks.")
    parser.add_argument("--highlight-words", action="store_true", help="Emit one timestamped cue per word in SRT/VTT output.")
    parser.add_argument("--max-words", type=int, default=20, help="Maximum words per subtitle cue.")
    parser.add_argument("--silence-gap", type=float, default=1.0, help="Split subtitle cues at this timing gap in seconds.")
    parser.add_argument("--max-duration", type=float, default=8.0, help="Maximum subtitle cue duration in seconds.")
    parser.add_argument("--device", default="cuda" if _has_mps() else "cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--memory-report",
        action="store_true",
        help="Record macOS process-footprint and MPS allocator memory in JSON output.",
    )
    parser.add_argument("--version", action="version", version="granite-reference 0.1.0")
    return parser


def _has_mps() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def main(argv: Iterable[str] | None = None) -> int:
    process_started = time.perf_counter()
    args = build_parser().parse_args(argv)
    missing = [str(path) for path in args.audio if not path.is_file()]
    if missing:
        print(f"Input file not found: {', '.join(missing)}", file=sys.stderr)
        return 2
    if args.chunk_duration < 0 or args.overlap_duration < 0:
        print("Chunk and overlap durations must be non-negative.", file=sys.stderr)
        return 2
    if args.chunk_duration and args.overlap_duration >= args.chunk_duration:
        print("Overlap duration must be smaller than chunk duration.", file=sys.stderr)
        return 2
    if args.max_words <= 0 or args.silence_gap < 0 or args.max_duration <= 0:
        print("Max words and max duration must be positive; silence gap must be non-negative.", file=sys.stderr)
        return 2

    try:
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            args.device = "mps" if _has_mps() else "cpu"
        model_load_started = time.perf_counter()
        model, processor = load_reference_model(args.model, args.device, args.precision)
        model_load_seconds = time.perf_counter() - model_load_started
        all_results = []

        for index, audio_path in enumerate(args.audio):
            audio_load_started = time.perf_counter()
            clip = load_audio(audio_path)
            audio_load_seconds = time.perf_counter() - audio_load_started
            if args.verbose:
                print(f"Transcribing {audio_path} ({clip.duration:.2f}s) ...", file=sys.stderr)
            if args.chunk_duration:
                result = transcribe_long_clip(
                    model,
                    processor,
                    clip,
                    args.device,
                    args.chunk_duration,
                    args.overlap_duration,
                )
            else:
                result = transcribe_clip(model, processor, clip, args.device)
            result["segments"] = segment_words(
                result.get("words", []),
                max_words=args.max_words,
                silence_gap=args.silence_gap,
                max_duration=args.max_duration,
            )
            result.update(
                {
                    "model": args.model,
                    "device": args.device,
                    "precision": args.precision,
                    "model_load_seconds": round(model_load_seconds, 6),
                    "audio_load_seconds": round(audio_load_seconds, 6),
                }
            )
            if args.memory_report:
                memory_probe_started = time.perf_counter()
                result["memory"] = memory_report(args.device)
                result["memory_probe_seconds"] = round(
                    time.perf_counter() - memory_probe_started, 6
                )
            result["elapsed_before_output_seconds"] = round(
                time.perf_counter() - process_started, 6
            )
            all_results.append(result)

            formats = ["txt", "srt", "vtt", "json"] if args.output_format == "all" else [args.output_format]
            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                for output_format in formats:
                    destination = output_path(
                        result, args.output_dir, args.output_template, output_format, index
                    )
                    destination.write_text(
                        output_result(result, output_format, args.highlight_words), encoding="utf-8"
                    )
                    if args.verbose:
                        print(f"Wrote {destination}", file=sys.stderr)
            elif len(all_results) == 1:
                if len(formats) != 1:
                    print("--output-format all requires --output-dir", file=sys.stderr)
                    return 2
                print(output_result(result, formats[0], args.highlight_words), end="")
            else:
                # Multiple stdout results are always JSON Lines for scripting.
                print(json.dumps(result, ensure_ascii=False))

        return 0
    except Exception as error:
        print(f"granite-reference: {error}", file=sys.stderr)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
