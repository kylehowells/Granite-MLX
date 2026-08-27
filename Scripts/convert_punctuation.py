#!/usr/bin/env python3
"""Convert the English punctuation/capitalization ONNX checkpoint to MLX."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import onnx
from onnx import numpy_helper
from sentencepiece import sentencepiece_model_pb2
from tokenizers import SentencePieceUnigramTokenizer

# tokenizers' SentencePiece importer expects the generated protobuf module at
# top level, while the sentencepiece wheel namespaces it inside its package.
sys.modules.setdefault("sentencepiece_model_pb2", sentencepiece_model_pb2)


MATRIX_IDS = [
    (947, 948, 951, 957, 958, 959),
    (960, 961, 964, 970, 971, 972),
    (973, 974, 977, 983, 984, 985),
    (986, 987, 990, 996, 997, 998),
    (999, 1000, 1003, 1009, 1010, 1011),
    (1012, 1013, 1016, 1022, 1023, 1024),
]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--bits", type=int, choices=(4, 5, 6, 8))
    parser.add_argument("--group-size", type=int, default=64, choices=(32, 64, 128))
    return parser.parse_args()


def semantic_weights(source: Path) -> dict[str, np.ndarray]:
    model = onnx.load(str(source / "punct_cap_seg_en.onnx"))
    raw = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    weights: dict[str, np.ndarray] = {
        "embeddings.word.weight": raw["bert_model.embeddings.word_embeddings.weight"],
        "embeddings.position.weight": raw["bert_model.embeddings.position_embeddings.weight"],
        "embeddings.token_type.weight": raw["bert_model.embeddings.token_type_embeddings.weight"],
        "embeddings.norm.weight": raw["bert_model.embeddings.LayerNorm.mod.weight"],
        "embeddings.norm.bias": raw["bert_model.embeddings.LayerNorm.mod.bias"],
        "decoder.punctuation_embedding.weight": raw["_decoder._punct_emb.weight"],
    }
    matrix_names = ("query", "key", "value", "attention_output", "intermediate", "output")
    for layer, ids in enumerate(MATRIX_IDS):
        base = f"bert_model.encoder.layer.{layer}"
        for name, matrix_id in zip(matrix_names, ids):
            # ONNX stores MatMul matrices [input, output]. MLX's linear/QMM
            # convention is [output, input].
            weights[f"layers.{layer}.{name}.weight"] = raw[f"onnx::MatMul_{matrix_id}"].T
        bias_sources = {
            "query": f"{base}.attention.self.query.bias",
            "key": f"{base}.attention.self.key.bias",
            "value": f"{base}.attention.self.value.bias",
            "attention_output": f"{base}.attention.output.dense.bias",
            "intermediate": f"{base}.intermediate.dense.bias",
            "output": f"{base}.output.dense.bias",
        }
        for name, source_name in bias_sources.items():
            weights[f"layers.{layer}.{name}.bias"] = raw[source_name]
        weights[f"layers.{layer}.attention_norm.weight"] = raw[f"{base}.attention.output.LayerNorm.mod.weight"]
        weights[f"layers.{layer}.attention_norm.bias"] = raw[f"{base}.attention.output.LayerNorm.mod.bias"]
        weights[f"layers.{layer}.output_norm.weight"] = raw[f"{base}.output.LayerNorm.mod.weight"]
        weights[f"layers.{layer}.output_norm.bias"] = raw[f"{base}.output.LayerNorm.mod.bias"]

    heads = {
        "post.0": (1025, "_decoder._punct_head_post._linears.0.bias"),
        "post.1": (1026, "_decoder._punct_head_post._linears.1.bias"),
        "pre.0": (1027, "_decoder._punct_head_pre._linears.0.bias"),
        "pre.1": (1028, "_decoder._punct_head_pre._linears.1.bias"),
        "seg.0": (1029, "_decoder._seg_head._linears.0.bias"),
        "seg.1": (1030, "_decoder._seg_head._linears.1.bias"),
        "cap.0": (1036, "_decoder._cap_head._linears.0.bias"),
        "cap.1": (1037, "_decoder._cap_head._linears.1.bias"),
    }
    for name, (matrix_id, bias_name) in heads.items():
        weights[f"decoder.{name}.weight"] = raw[f"onnx::MatMul_{matrix_id}"].T
        weights[f"decoder.{name}.bias"] = raw[bias_name]
    return weights


def main() -> None:
    args = parse_args()
    source_model = args.source / "punct_cap_seg_en.onnx"
    values = semantic_weights(args.source)
    output: dict[str, mx.array] = {}
    quantized: list[str] = []
    for name, value in values.items():
        array = mx.array(value.astype(np.float16))
        if args.bits and name.endswith(".weight") and array.ndim == 2 and array.shape[-1] % args.group_size == 0:
            packed, scales, biases = mx.quantize(
                array, group_size=args.group_size, bits=args.bits, mode="affine"
            )
            output[name] = packed
            prefix = name.removesuffix(".weight")
            output[f"{prefix}.scales"] = scales
            output[f"{prefix}.biases"] = biases
            quantized.append(name)
        else:
            output[name] = array
    mx.eval(*output.values())

    args.destination.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.destination / "model.safetensors"), output, metadata={"format": "mlx"})
    for filename in ("spe_32k_lc_en.model", "config.yaml", "README.md"):
        shutil.copy2(args.source / filename, args.destination / filename)
    tokenizer = SentencePieceUnigramTokenizer.from_spm(
        str(args.source / "spe_32k_lc_en.model")
    )
    tokenizer.save(str(args.destination / "tokenizer.json"))
    tokenizer_config = {
        "tokenizer_class": "XLMRobertaTokenizer",
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "model_max_length": 256,
    }
    (args.destination / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2) + "\n", encoding="utf-8"
    )
    config = {
        "architecture": "bert-punctuation-capitalization-segmentation",
        "hidden_size": 512,
        "intermediate_size": 2048,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "max_length": 256,
        "layer_norm_epsilon": 1e-12,
        "precision": "fp16",
        "quantization": None if args.bits is None else {
            "bits": args.bits,
            "group_size": args.group_size,
            "mode": "affine",
            "quantized_tensors": quantized,
        },
        "source": {
            "model_id": "1-800-BAD-CODE/punctuation_fullstop_truecase_english",
            "revision": "b26fd1c40e88678859048898218ea4edcc24c84a",
            "onnx_sha256": digest(source_model),
        },
    }
    (args.destination / "mlx_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"destination": str(args.destination), "tensor_count": len(output), **config}, indent=2))


if __name__ == "__main__":
    main()
