# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from transformers import Pretrainedconfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(LlamaRotaryEmbedding):
    """Rotary Position Embedding Layer.

    This class extends the LlamaRotaryEmbedding to provide a rotary embedding layer that can be used in attention
    mechanisms.
    """

    def __init__(self, config: Pretrainedconfig):
        super().__init__(config)

    def forward(self, position_ids, query, key, hidden_states):
        """Forward pass for the Rotary Position Embedding layer."""
        cos, sin = super().forward(hidden_states, position_ids)

        return apply_rotary_pos_emb(query, key, cos, sin)
