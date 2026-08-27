"""Conversion-oriented PyTorch implementation of Granite Speech 5.0 CTC.

The model uses only explicit tensor operations that Core ML Tools can lower.
Its module tree intentionally matches the published checkpoint, allowing a
strict state-dict load without depending on a particular Transformers release.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F


class GraniteFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.silu(self.linear1(hidden)))


class GraniteBlockAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dimension: int,
        context_size: int,
        max_position_embeddings: int,
    ):
        super().__init__()
        inner = num_heads * head_dimension
        self.num_heads = num_heads
        self.head_dimension = head_dimension
        self.context_size = context_size
        self.scale = head_dimension**-0.5
        self.q_proj = nn.Linear(hidden_size, inner, bias=False)
        self.k_proj = nn.Linear(hidden_size, inner, bias=False)
        self.v_proj = nn.Linear(hidden_size, inner, bias=False)
        self.o_proj = nn.Linear(inner, hidden_size)
        self.rel_pos_emb = nn.Embedding(
            2 * max_position_embeddings + 1, head_dimension)
        positions = torch.arange(context_size, dtype=torch.int64)
        distances = positions[:, None] - positions[None, :]
        self.register_buffer(
            "attention_distances",
            distances.clamp(-context_size, context_size) + max_position_embeddings,
            persistent=False,
        )

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_length: int,
    ) -> torch.Tensor:
        batch, length, _ = query.shape
        block_count = length // block_length

        def shape(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch, block_count, block_length,
                self.num_heads, self.head_dimension,
            ).permute(0, 1, 3, 2, 4)

        query = shape(query)
        key = shape(key)
        value = shape(value)
        relative = self.rel_pos_emb(
            self.attention_distances[:block_length, :block_length])
        relative_bias = torch.einsum(
            "bmhcd,crd->bmhcr", query, relative) * self.scale
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        weights = torch.softmax(scores + relative_bias, dim=-1)
        attended = torch.matmul(weights, value)
        return attended.permute(0, 1, 3, 2, 4).reshape(batch, length, -1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        length = hidden.shape[1]
        full_length = (length // self.context_size) * self.context_size
        query = self.q_proj(hidden)
        key = self.k_proj(hidden)
        value = self.v_proj(hidden)
        outputs: list[torch.Tensor] = []
        if full_length:
            outputs.append(self._attend(
                query[:, :full_length], key[:, :full_length],
                value[:, :full_length], self.context_size))
        if full_length < length:
            outputs.append(self._attend(
                query[:, full_length:], key[:, full_length:],
                value[:, full_length:], length - full_length))
        attended = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
        return self.o_proj(attended)


class GraniteConvolution(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expansion_factor: int,
        kernel_size: int,
        stride: int,
    ):
        super().__init__()
        inner = hidden_size * expansion_factor
        self.kernel_size = kernel_size
        self.pointwise_lin1 = nn.Linear(hidden_size, inner * 2)
        self.depthwise_conv = nn.Conv1d(
            inner, inner, kernel_size, stride=stride,
            padding=0, groups=inner, bias=False)
        self.norm = nn.BatchNorm1d(inner, eps=1e-5)
        self.pointwise_lin2 = nn.Linear(inner, hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        first, gate = self.pointwise_lin1(hidden).chunk(2, dim=-1)
        hidden = first * torch.sigmoid(gate)
        left = self.kernel_size // 2
        right = left - (self.kernel_size + 1) % 2
        hidden = F.pad(hidden.transpose(1, 2), (left, right))
        hidden = self.depthwise_conv(hidden)
        hidden = F.silu(self.norm(hidden)).transpose(1, 2)
        return self.pointwise_lin2(hidden)


class GraniteConformerLayer(nn.Module):
    def __init__(self, config: dict, subsamples: bool):
        super().__init__()
        hidden = config["hidden_size"]
        intermediate = config["intermediate_size"]
        self.subsamples = subsamples
        self.norm_feed_forward1 = nn.LayerNorm(hidden, eps=1e-5)
        self.feed_forward1 = GraniteFeedForward(hidden, intermediate)
        self.norm_self_att = nn.LayerNorm(hidden, eps=1e-5)
        self.self_attn = GraniteBlockAttention(
            hidden, config["num_attention_heads"], config["head_dim"],
            config["context_size"], config["max_position_embeddings"])
        self.norm_conv = nn.LayerNorm(hidden, eps=1e-5)
        self.conv = GraniteConvolution(
            hidden, config["conv_expansion_factor"],
            config["conv_kernel_size"], 2 if subsamples else 1)
        self.norm_feed_forward2 = nn.LayerNorm(hidden, eps=1e-5)
        self.feed_forward2 = GraniteFeedForward(hidden, intermediate)
        self.norm_out = nn.LayerNorm(hidden, eps=1e-5)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        hidden = source + 0.5 * self.feed_forward1(
            self.norm_feed_forward1(source))
        hidden = hidden + self.self_attn(self.norm_self_att(hidden))
        convolution = self.conv(self.norm_conv(hidden))
        if self.subsamples:
            half = hidden.shape[1] // 2
            hidden = hidden[:, : half * 2].reshape(
                hidden.shape[0], half, 2, hidden.shape[2]).mean(dim=2)
            hidden = hidden + convolution[:, :half]
        else:
            hidden = hidden + convolution
        hidden = hidden + 0.5 * self.feed_forward2(
            self.norm_feed_forward2(hidden))
        return self.norm_out(hidden)


class GraniteEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden = config["hidden_size"]
        vocabulary = config["vocab_size"]
        self.input_linear = nn.Linear(320, hidden)
        subsampling = set(config["subsample_layers"])
        self.layers = nn.ModuleList([
            GraniteConformerLayer(config, index in subsampling)
            for index in range(config["num_hidden_layers"])
        ])
        self.out = nn.Linear(hidden, vocabulary)
        self.out_mid = nn.Linear(vocabulary, hidden)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_linear(features)
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index + 1 == len(self.layers) // 2:
                probabilities = torch.softmax(self.out(hidden).float(), dim=-1)
                hidden = hidden + self.out_mid(probabilities.to(hidden.dtype))
        return hidden

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.out(self.encode(features))


class GraniteCoreMLModel(nn.Module):
    """Full encoder returning one greedy CTC token ID per output frame."""

    def __init__(self, config: dict):
        super().__init__()
        self.encoder = GraniteEncoder(config)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.encoder(features), dim=-1).to(torch.int32)


class GraniteCoreMLHiddenModel(nn.Module):
    """Encoder variant returning final hidden states before the CTC head."""

    def __init__(self, encoder: GraniteEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(features)


def load_granite_coreml_model(model_directory: Path) -> GraniteCoreMLModel:
    config = json.loads((model_directory / "config.json").read_text())["encoder_config"]
    model = GraniteCoreMLModel(config)
    state = load_file(model_directory / "model.safetensors")
    state = {
        key: value.float()
        for key, value in state.items()
        if not key.endswith("num_batches_tracked")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [name for name in missing if not name.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/module mismatch; missing={missing}, unexpected={unexpected}")
    model.eval()
    return model
