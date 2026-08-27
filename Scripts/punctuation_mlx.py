"""MLX inference runtime for the converted punctuation model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from punctuators.collectors.pcs_collector import PunctCapSegResultCollector
from punctuators.data.infer_dataset import TextInferenceDataset
from sentencepiece import SentencePieceProcessor
from torch.utils.data import DataLoader


class MLXPunctuationModel:
    def __init__(self, directory: Path):
        self.directory = directory
        self.config = json.loads((directory / "mlx_config.json").read_text())
        self.weights = mx.load(str(directory / "model.safetensors"))
        self.quantization = self.config.get("quantization")
        self.tokenizer = SentencePieceProcessor(str(directory / "spe_32k_lc_en.model"))
        mx.eval(*self.weights.values())

    def _linear(self, x: mx.array, prefix: str) -> mx.array:
        weight = self.weights[f"{prefix}.weight"]
        scales = self.weights.get(f"{prefix}.scales")
        if scales is None:
            result = x @ weight.T
        else:
            q = self.quantization
            result = mx.quantized_matmul(
                x, weight, scales, self.weights[f"{prefix}.biases"],
                transpose=True, group_size=q["group_size"], bits=q["bits"], mode=q["mode"],
            )
        bias = self.weights.get(f"{prefix}.bias")
        return result if bias is None else result + bias

    def _embedding(self, ids: mx.array, prefix: str) -> mx.array:
        weight = self.weights[f"{prefix}.weight"]
        scales = self.weights.get(f"{prefix}.scales")
        if scales is None:
            return weight[ids]
        q = self.quantization
        return mx.dequantize(
            weight[ids], scales[ids], self.weights[f"{prefix}.biases"][ids],
            group_size=q["group_size"], bits=q["bits"], mode=q["mode"],
        )

    def _norm(self, x: mx.array, prefix: str) -> mx.array:
        return mx.fast.layer_norm(
            x, self.weights[f"{prefix}.weight"], self.weights[f"{prefix}.bias"], 1e-12
        )

    @staticmethod
    def _gelu(x: mx.array) -> mx.array:
        return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))

    def __call__(self, input_ids: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        batch, length = input_ids.shape
        positions = mx.arange(length)[None, :]
        token_types = mx.zeros((batch, length), dtype=mx.int32)
        hidden = (
            self._embedding(input_ids, "embeddings.word")
            + self._embedding(positions, "embeddings.position")
            + self._embedding(token_types, "embeddings.token_type")
        )
        hidden = self._norm(hidden, "embeddings.norm")
        valid = input_ids != 3

        for layer in range(6):
            prefix = f"layers.{layer}"
            query = self._linear(hidden, f"{prefix}.query").reshape(batch, length, 8, 64).transpose(0, 2, 1, 3)
            key = self._linear(hidden, f"{prefix}.key").reshape(batch, length, 8, 64).transpose(0, 2, 1, 3)
            value = self._linear(hidden, f"{prefix}.value").reshape(batch, length, 8, 64).transpose(0, 2, 1, 3)
            scores = (query @ key.transpose(0, 1, 3, 2)) / 8.0
            scores = mx.where(valid[:, None, None, :], scores, mx.array(-10000.0, dtype=scores.dtype))
            context = (mx.softmax(scores, axis=-1) @ value).transpose(0, 2, 1, 3).reshape(batch, length, 512)
            attention = self._linear(context, f"{prefix}.attention_output")
            hidden = self._norm(hidden + attention, f"{prefix}.attention_norm")
            intermediate = self._gelu(self._linear(hidden, f"{prefix}.intermediate"))
            output = self._linear(intermediate, f"{prefix}.output")
            hidden = self._norm(hidden + output, f"{prefix}.output_norm")

        post = self._linear(mx.maximum(self._linear(hidden, "decoder.post.0"), 0), "decoder.post.1")
        pre = self._linear(mx.maximum(self._linear(hidden, "decoder.pre.0"), 0), "decoder.pre.1")
        post_ids = mx.argmax(post, axis=-1)
        punct = self.weights["decoder.punctuation_embedding.weight"][post_ids]
        seg = self._linear(mx.maximum(self._linear(mx.concatenate([hidden, punct], axis=-1), "decoder.seg.0"), 0), "decoder.seg.1")
        seg_ids = mx.argmax(seg, axis=-1)
        shifted_seg = mx.concatenate([mx.zeros((batch, 1), dtype=seg_ids.dtype), seg_ids[:, :-1]], axis=1)[..., None].astype(hidden.dtype)
        cap = self._linear(mx.maximum(self._linear(mx.concatenate([hidden, shifted_seg], axis=-1), "decoder.cap.0"), 0), "decoder.cap.1")
        return mx.argmax(pre, axis=-1), post_ids, cap > 0, seg_ids

    def infer(self, texts: list[str], batch_size_tokens: int = 4096, overlap: int = 16) -> list[list[str]]:
        collectors = [
            PunctCapSegResultCollector(
                sp_model=self.tokenizer, apply_sbd=True, overlap=overlap
            )
            for _ in texts
        ]
        dataset = TextInferenceDataset(
            texts=texts, batch_size_tokens=batch_size_tokens, overlap=overlap,
            max_length=256, spe_model_path=str(self.directory / "spe_32k_lc_en.model"),
        )
        loader = DataLoader(dataset, num_workers=0, collate_fn=dataset.collate_fn, batch_sampler=dataset.sampler)
        pre_labels = [None, "¿"]
        post_labels = [None, "<ACRONYM>", ".", ",", "?"]
        for input_ids, batch_indices, input_indices, lengths in loader:
            ids = mx.array(input_ids.numpy().astype(np.int32))
            pre, post, cap, seg = self(ids)
            mx.eval(pre, post, cap, seg)
            pre_np, post_np, cap_np, seg_np = map(np.array, (pre, post, cap, seg))
            for i in range(input_ids.shape[0]):
                length = int(lengths[i])
                sl = slice(1, length - 1)
                collectors[int(batch_indices[i])].collect(
                    ids=input_ids[i, sl].tolist(),
                    pre_preds=[pre_labels[x] for x in pre_np[i, sl]],
                    post_preds=[post_labels[x] for x in post_np[i, sl]],
                    cap_preds=cap_np[i, sl].tolist(),
                    sbd_preds=seg_np[i, sl].tolist(),
                    idx=int(input_indices[i]),
                )
        return [collector.produce() for collector in collectors]
