import math
from typing import List, Optional, Tuple

from base_counter import BaseCounter

DEFAULT_SEQ_LEN = 1024
ACTIVATION_BYTES = 2
INPUT_ID_SIZE = 4  # input IDs are 4 bytes each
EMBEDDING_LMHEAD_PRECISION_BYTES = 2  # LM head is typically in fp16
GATE_PRECISION = 2
ATTN_PRECISION = 2
QUANTIZATION_BYTES = 4  # float 32 for quantization block


class MemCounter(BaseCounter):
    """Base counter for memory access bytes calculations. Support MHA, GQA, MLA with LoRA"""

    def __init__(self, **kwargs):
        super().__init__()
        self.model_precision_bytes = kwargs.get("precision_bytes", 2)

    def tensor_size(
        self,
        tensors: List[Tuple[List[int], int]],
        quantization_block_size: Optional[int] = 1,
    ) -> float:
        """Calculate total memory size for a list of tensors."""
        if not tensors:
            return 0.0
        size_sum = 0
        for tensor_info in tensors:
            if isinstance(tensor_info, tuple):
                dimensions, bytes_per_element = tensor_info[0], tensor_info[1]
            elif isinstance(tensor_info, list):
                dimensions = tensor_info
                bytes_per_element = self.model_precision_bytes
            if not dimensions or any(dim <= 0 for dim in dimensions):
                raise ValueError(f"Invalid dimensions: {dimensions}")
            size_sum += math.prod(dimensions) * bytes_per_element
            if bytes_per_element == 1:
                scaled_block_dims = [
                    max(1, d // quantization_block_size) for d in dimensions
                ]
                size_sum += math.prod(scaled_block_dims) * QUANTIZATION_BYTES
        return size_sum

    @property
    def unit(self) -> float:
        """Return the unit of measurement (terabytes)."""
        return 1e12

    def _get_seq_len_for_phase(self, **kwargs) -> int:
        """Get sequence length based on the phase (prefill vs decode)."""
        raise NotImplementedError("Subclasses must implement this method")

    def embedding_lmhead(self, vocab_size: int, hidden_size: int, **kwargs) -> float:
        """Calculate memory access for embedding and LM head."""
        seq_len = self._get_seq_len_for_phase(**kwargs)

        embedding_tensors = [
            ([seq_len], INPUT_ID_SIZE),  # input_ids
            ([vocab_size, hidden_size], EMBEDDING_LMHEAD_PRECISION_BYTES),  # embeddings
            ([seq_len, hidden_size], EMBEDDING_LMHEAD_PRECISION_BYTES),  # hidden_states
        ]

        lm_head_tensors = [
            ([seq_len, hidden_size], EMBEDDING_LMHEAD_PRECISION_BYTES),  # hidden_states
            ([hidden_size, vocab_size], EMBEDDING_LMHEAD_PRECISION_BYTES),  # lm_head
            ([seq_len, vocab_size], EMBEDDING_LMHEAD_PRECISION_BYTES),  # logits
        ]

        return self.tensor_size(embedding_tensors) + self.tensor_size(lm_head_tensors)

    def gateup_mlp(self, hidden_size: int, intermediate_size: int, **kwargs) -> float:
        """Calculate memory access for gate-up MLP layer."""
        seq_len = self._get_seq_len_for_phase(**kwargs)
        gateup_proj_tensors = [
            ([seq_len, hidden_size]),  # input
            ([hidden_size, intermediate_size * 2]),  # gateup proj weight
            ([seq_len, intermediate_size * 2], ACTIVATION_BYTES),  # output
        ]

        activation_tensors = [
            ([seq_len, intermediate_size * 2], ACTIVATION_BYTES),  # input
            ([seq_len, intermediate_size], ACTIVATION_BYTES),  # activation output
        ]

        down_proj_tensors = [
            ([seq_len, intermediate_size]),  # input
            ([intermediate_size, hidden_size]),  # down proj weight
            ([seq_len, hidden_size], ACTIVATION_BYTES),  # output
        ]

        return (
            self.tensor_size(gateup_proj_tensors)
            + self.tensor_size(activation_tensors)
            + self.tensor_size(down_proj_tensors)
        )

    def moe(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        n_shared_experts: int = 0,
        **kwargs,
    ) -> float:
        """Calculate memory access for Mixture of Experts (MoE) layer.    Total memory access in bytes"""
        seq_len = self._get_seq_len_for_phase(**kwargs)
        num_experts_per_tok = kwargs.get("num_experts_per_tok", 2) + n_shared_experts
        effective_seq_len = seq_len * num_experts_per_tok

        gate_tensors = [
            ([seq_len, hidden_size], GATE_PRECISION),  # input
            ([hidden_size, num_experts], GATE_PRECISION),  # moe gate weight
            ([seq_len, num_experts], ACTIVATION_BYTES),  # output
        ]

        # Number of experts actually used
        experts_used = min(num_experts, effective_seq_len)

        gateup_proj_tensors = [
            ([effective_seq_len, hidden_size]),  # input
            (
                [experts_used, hidden_size, moe_intermediate_size * 2]
            ),  # moe gateup proj weight
            (
                [effective_seq_len, moe_intermediate_size * 2],
                ACTIVATION_BYTES,
            ),  # output
        ]

        activation_tensors = [
            ([effective_seq_len, moe_intermediate_size * 2], ACTIVATION_BYTES),  # input
            (
                [effective_seq_len, moe_intermediate_size],
                ACTIVATION_BYTES,
            ),  # activation output
        ]

        down_proj_tensors = [
            ([effective_seq_len, moe_intermediate_size]),  # input
            ([experts_used, moe_intermediate_size, hidden_size]),  # down proj weight
            ([effective_seq_len, hidden_size], ACTIVATION_BYTES),  # output
        ]

        return (
            self.tensor_size(gate_tensors)
            + self.tensor_size(gateup_proj_tensors)
            + self.tensor_size(activation_tensors)
            + self.tensor_size(down_proj_tensors)
        )

    def attention(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        num_key_value_heads: Optional[int],
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        q_lora_rank: int = 0,
        kv_lora_rank: int = 0,
        **kwargs,
    ) -> float:
        """Calculate memory access for attention layer."""
        seq_len = self._get_seq_len_for_phase(**kwargs)
        cache_len = kwargs.get("seq_len", DEFAULT_SEQ_LEN)  # KV cache length
        # QKV projections
        qkv_proj_mem = self._qkv_proj(
            seq_len,
            hidden_size,
            num_attention_heads,
            head_dim,
            num_key_value_heads,
            qk_rope_head_dim,
            qk_nope_head_dim,
            q_lora_rank,
            kv_lora_rank,
        )

        # Attention computation
        attn_mem = self._attn(
            seq_len,
            cache_len,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            qk_rope_head_dim,
            qk_nope_head_dim,
            kv_lora_rank,
        )

        # Output projection
        o_proj_tensors = [
            ([seq_len, num_attention_heads * head_dim]),  # attn output
            ([num_attention_heads * head_dim, hidden_size]),  # o_proj weight
            ([seq_len, hidden_size], ACTIVATION_BYTES),  # output
        ]

        return qkv_proj_mem + attn_mem + self.tensor_size(o_proj_tensors)

    def _qkv_proj(
        self,
        seq_len: int,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        num_key_value_heads: Optional[int],
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
    ) -> float:
        """Calculate memory access for QKV projections in attention."""
        if qk_rope_head_dim == 0:
            # MHA and GQA
            return self._gqa_mha_qkv_proj(
                seq_len, hidden_size, num_attention_heads, head_dim, num_key_value_heads
            )
        else:
            # MLA with LoRA
            return self._mla_qkv_proj(
                seq_len,
                hidden_size,
                num_attention_heads,
                head_dim,
                qk_rope_head_dim,
                qk_nope_head_dim,
                q_lora_rank,
                kv_lora_rank,
            )

    def _gqa_mha_qkv_proj(
        self,
        seq_len: int,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        num_key_value_heads: Optional[int],
    ) -> float:
        """Calculate memory access for standard QKV projections (MHA/GQA)."""
        if num_key_value_heads is None:
            qkv_size = hidden_size * 3  # MHA
        else:
            qkv_size = (
                num_attention_heads * head_dim + num_key_value_heads * head_dim * 2
            )  # GQA

        qkv_proj_tensors = [
            ([seq_len, hidden_size]),  # input
            ([hidden_size, qkv_size]),  # qkv_proj weight
            ([seq_len, qkv_size], ACTIVATION_BYTES),  # output
        ]
        return self.tensor_size(qkv_proj_tensors)

    def _mla_qkv_proj(
        self,
        seq_len: int,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
    ) -> float:
        """Calculate memory access for MLA QKV projections."""
        fused_qkv_tensors = [
            ([seq_len, hidden_size]),  # input
            (
                [hidden_size, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
            ),  # fused_qkv_proj weight
            (
                [seq_len, q_lora_rank + kv_lora_rank + qk_rope_head_dim],
                ACTIVATION_BYTES,
            ),  # output
        ]

        q_b_proj_tensors = [
            ([seq_len, q_lora_rank]),  # input
            (
                [
                    q_lora_rank,
                    num_attention_heads * (qk_rope_head_dim + qk_nope_head_dim),
                ]
            ),  # q_b_proj weight
            (
                [seq_len, num_attention_heads * (qk_rope_head_dim + qk_nope_head_dim)],
                ACTIVATION_BYTES,
            ),  # output
        ]

        mem = self.tensor_size(fused_qkv_tensors) + self.tensor_size(q_b_proj_tensors)
        mem += self._mla_kv_proj(
            seq_len, kv_lora_rank, num_attention_heads, head_dim, qk_nope_head_dim
        )
        return mem

    def _mla_kv_proj(
        self,
        seq_len: int,
        kv_lora_rank: int,
        num_attention_heads: int,
        head_dim: int,
        qk_nope_head_dim: int,
    ) -> float:
        """Calculate memory access for MLA KV projection - to be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")

    def _attn(
        self,
        seq_len: int,
        cache_len: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int],
        head_dim: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
    ) -> float:
        """Calculate memory access for attention computation."""
        if qk_rope_head_dim == 0:
            # Standard attention (GQA and MHA)
            if num_key_value_heads is None:
                # MHA
                attn_tensors = [
                    ([seq_len, num_attention_heads * head_dim], ATTN_PRECISION),  # Q
                    (
                        [seq_len + cache_len, num_attention_heads * head_dim],
                        ATTN_PRECISION,
                    ),  # K
                    (
                        [seq_len + cache_len, num_attention_heads * head_dim],
                        ATTN_PRECISION,
                    ),  # V
                    (
                        [seq_len, num_attention_heads * head_dim],
                        ATTN_PRECISION,
                    ),  # output
                ]
            else:
                # GQA
                attn_tensors = [
                    ([seq_len, num_attention_heads * head_dim], ATTN_PRECISION),  # Q
                    (
                        [seq_len + cache_len, num_key_value_heads * head_dim],
                        ATTN_PRECISION,
                    ),  # K
                    (
                        [seq_len + cache_len, num_key_value_heads * head_dim],
                        ATTN_PRECISION,
                    ),  # V
                    (
                        [seq_len, num_attention_heads * head_dim],
                        ATTN_PRECISION,
                    ),  # output
                ]
            return self.tensor_size(attn_tensors)
        else:
            # MLA - this is where prefill and decode differ
            return self._mla_attention(
                seq_len,
                cache_len,
                num_attention_heads,
                qk_rope_head_dim,
                qk_nope_head_dim,
                kv_lora_rank,
            )

    def _mla_attention(
        self,
        seq_len: int,
        cache_len: int,
        num_attention_heads: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
    ) -> float:
        """Get attention dimension for MLA - to be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")


class PrefillMemCounter(MemCounter):
    """Memory counter for the prefill phase of transformer inference."""

    @property
    def metric_name(self) -> str:
        """Return the metric name for prefill memory usage."""
        return "Prefill-Mem(TB)"

    def _get_seq_len_for_phase(self, **kwargs) -> int:
        """For prefill, use the full sequence length."""
        return kwargs.get("seq_len", DEFAULT_SEQ_LEN)

    def _mla_kv_proj(
        self,
        seq_len: int,
        kv_lora_rank: int,
        num_attention_heads: int,
        head_dim: int,
        qk_nope_head_dim: int,
    ) -> float:
        """Calculate memory access for MLA KV projection in prefill phase."""
        kv_proj_tensors = [
            ([seq_len, kv_lora_rank]),  # input
            (
                [kv_lora_rank, num_attention_heads * (head_dim + qk_nope_head_dim)]
            ),  # kv_b_proj weight
            (
                [seq_len, num_attention_heads * (head_dim + qk_nope_head_dim)],
                ACTIVATION_BYTES,
            ),  # output
        ]
        return self.tensor_size(kv_proj_tensors)

    def _mla_attention(
        self,
        seq_len: int,
        cache_len: int,
        num_attention_heads: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
    ) -> float:
        """Get attention dimension for MLA in prefill phase."""
        attn = [
            (
                [seq_len, num_attention_heads * (qk_rope_head_dim + qk_nope_head_dim)],
                ATTN_PRECISION,
            ),  # Q
            (
                [seq_len, num_attention_heads * (qk_rope_head_dim + qk_nope_head_dim)],
                ATTN_PRECISION,
            ),  # K
            ([seq_len, num_attention_heads * qk_nope_head_dim], ATTN_PRECISION),  # V
            (
                [seq_len, num_attention_heads * qk_nope_head_dim],
                ATTN_PRECISION,
            ),  # output
        ]
        return self.tensor_size(attn)


class DecodeMemCounter(MemCounter):
    """Memory counter for the decode phase of transformer inference."""

    @property
    def metric_name(self) -> str:
        """Return the metric name for decode memory usage."""
        return "Decode-Mem(TB)"

    def _get_seq_len_for_phase(self, **kwargs) -> int:
        """For decode, process only one token at a time."""
        return 1

    def _mla_kv_proj(
        self,
        seq_len: int,
        kv_lora_rank: int,
        num_attention_heads: int,
        head_dim: int,
        qk_nope_head_dim: int,
    ) -> float:
        """Calculate memory access for MLA KV projection in decode phase."""
        q_nope_tensors = [
            ([seq_len, num_attention_heads * qk_nope_head_dim]),  # input
            ([num_attention_heads * qk_nope_head_dim, kv_lora_rank]),  # w_kc
            ([seq_len, kv_lora_rank], ACTIVATION_BYTES),  # q_nope output
        ]

        bmm_tensors = [
            ([seq_len, kv_lora_rank, num_attention_heads]),  # input
            ([kv_lora_rank, num_attention_heads, head_dim]),  # w_vc
            (
                [seq_len, num_attention_heads * head_dim],
                ACTIVATION_BYTES,
            ),  # attn_bmm output
        ]

        return self.tensor_size(q_nope_tensors) + self.tensor_size(bmm_tensors)

    def _mla_attention(
        self,
        seq_len: int,
        cache_len: int,
        num_attention_heads: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
    ) -> float:
        """Get attention dimension for MLA in decode phase."""
        attn = [
            (
                [seq_len, num_attention_heads * (kv_lora_rank + qk_rope_head_dim)],
                ATTN_PRECISION,
            ),  # Q
            ([cache_len, (kv_lora_rank + qk_rope_head_dim)], ATTN_PRECISION),  # K
            ([seq_len, num_attention_heads * kv_lora_rank], ATTN_PRECISION),  # output
        ]
        return self.tensor_size(attn)
