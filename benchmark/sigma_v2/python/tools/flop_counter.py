
from typing import Optional
from base_counter import BaseCounter

DEFAULT_SEQ_LEN = 1024

class FlopCounter(BaseCounter):
    """Base counter for FLOP (Floating Point Operations) calculations. Support MHA, GQA, MLA with LoRA """
    
    @staticmethod
    def gemm_flops(m: int, n: int, k: int) -> float:
        return 2.0 * m * n * k
    
    @property
    def unit(self) -> float:
        return 1e12  # Using trillions as the unit for FLOPs
    
    def _get_seq_len_for_phase(self, **kwargs) -> int:
        """Get sequence length based on the phase (prefill vs decode)."""
        raise NotImplementedError("Subclasses must implement this method")

    def embedding_lmhead(self, vocab_size: int, hidden_size: int, **kwargs) -> float:
        """Calculate FLOPs for embedding and LM head."""
        seq_len = self._get_seq_len_for_phase(**kwargs)
        return self.gemm_flops(seq_len, hidden_size, vocab_size)

    def gateup_mlp(self, hidden_size: int, intermediate_size: int, **kwargs) -> float:
        """Calculate FLOPs for gate-up MLP layer."""
        seq_len = self._get_seq_len_for_phase(**kwargs)
        return (
            self.gemm_flops(seq_len, hidden_size, intermediate_size * 2) +  # gateup proj
            self.gemm_flops(seq_len, intermediate_size, hidden_size) +      # down proj
            seq_len * intermediate_size  # activation function
        )

    def moe(self, hidden_size: int, 
            moe_intermediate_size: int, 
            num_experts: int, 
            n_shared_experts: int = 0, 
            **kwargs) -> float:
        """Calculate FLOPs for Mixture of Experts (MoE) layer."""
        seq_len = self._get_seq_len_for_phase(**kwargs)
        num_experts_per_tok = kwargs.get('num_experts_per_tok', 2) + n_shared_experts
        effective_seq_len = seq_len * num_experts_per_tok
        
        return (
            self.gemm_flops(seq_len, hidden_size, num_experts) +  # moe gate
            self.gemm_flops(effective_seq_len, hidden_size, moe_intermediate_size * 2) +  # gateup proj
            self.gemm_flops(effective_seq_len, moe_intermediate_size, hidden_size) +      # down proj
            effective_seq_len * moe_intermediate_size  # activation function
        )
        
    def attention(self, hidden_size: int, num_attention_heads: int, head_dim: int,
                 num_key_value_heads: Optional[int], qk_rope_head_dim: int, qk_nope_head_dim: int, 
                 q_lora_rank: int = 0, kv_lora_rank: int = 0, **kwargs) -> float:
        """Calculate FLOPs for attention layer."""
        seq_len = self._get_seq_len_for_phase(**kwargs)
        cache_len = kwargs.get('seq_len', DEFAULT_SEQ_LEN)  # KV cache length
        
        # QKV projections
        qkv_proj = self._qkv_proj(
            seq_len, hidden_size, num_attention_heads, head_dim,
            num_key_value_heads, qk_rope_head_dim, qk_nope_head_dim,
            q_lora_rank, kv_lora_rank
        )
        attn = self._attn(
            seq_len, cache_len, num_attention_heads, head_dim,
            qk_rope_head_dim, qk_nope_head_dim, kv_lora_rank
        )
        o_proj = self.gemm_flops(seq_len, num_attention_heads * head_dim, hidden_size)
        return qkv_proj + attn + o_proj
    
    def _qkv_proj(self, seq_len: int, 
                 hidden_size: int, 
                 num_attention_heads: int, 
                 head_dim: int,
                 num_key_value_heads: Optional[int],
                 qk_rope_head_dim: int, 
                 qk_nope_head_dim: int,
                 q_lora_rank: int, 
                 kv_lora_rank: int) -> float:
        """Calculate FLOPs for QKV projections in attention."""
        if qk_rope_head_dim == 0:
            # MHA and GQA
            return self._gqa_mha_qkv_proj(seq_len, hidden_size, num_attention_heads, 
                                         head_dim, num_key_value_heads)
        else:
            # MLA with LoRA
            return self._mla_qkv_proj(seq_len, hidden_size, num_attention_heads,
                                    head_dim, qk_rope_head_dim, qk_nope_head_dim,
                                    q_lora_rank, kv_lora_rank)
    
    def _gqa_mha_qkv_proj(self, seq_len: int, hidden_size: int, num_attention_heads: int,
                          head_dim: int, num_key_value_heads: Optional[int]) -> float:
        """Calculate FLOPs for standard QKV projections (MHA/GQA)."""
        if num_key_value_heads is None:
            qkv_size = hidden_size * 3  # MHA
        else:
            qkv_size = num_attention_heads * head_dim + num_key_value_heads * head_dim * 2  # GQA
        return self.gemm_flops(seq_len, hidden_size, qkv_size)
    
    def _mla_qkv_proj(self, seq_len: int, hidden_size: int, num_attention_heads: int,
                     head_dim: int, qk_rope_head_dim: int, qk_nope_head_dim: int,
                     q_lora_rank: int, kv_lora_rank: int) -> float:
        """Calculate FLOPs for Multi-Latent Attention (MLA) QKV projections."""
        flops = self.gemm_flops(seq_len, hidden_size, q_lora_rank + kv_lora_rank + qk_rope_head_dim)
        flops += self.gemm_flops(seq_len, q_lora_rank, 
                               num_attention_heads * (qk_rope_head_dim + qk_nope_head_dim))
        flops += self._mla_kv_proj(seq_len, kv_lora_rank, num_attention_heads, 
                                 head_dim, qk_nope_head_dim)
        return flops
    
    def _mla_kv_proj(self, seq_len: int, kv_lora_rank: int, num_attention_heads: int,
                    head_dim: int, qk_nope_head_dim: int) -> float:
        """Calculate FLOPs for MLA KV projection - to be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def _attn(self, seq_len: int, cache_len: int,
             num_attention_heads: int, head_dim: int,
             qk_rope_head_dim: int, qk_nope_head_dim: int,
             kv_lora_rank: int) -> float:
        """Calculate FLOPs for attention computation."""
        if qk_rope_head_dim == 0:
            # Standard attention (GQA and MHA)
            attention_dim = num_attention_heads * head_dim
        else:
            # MLA - this is where prefill and decode differ
            attention_dim = self._get_mla_attention_dim(num_attention_heads, qk_rope_head_dim, 
                                                      qk_nope_head_dim, kv_lora_rank)
        
        return self.gemm_flops(seq_len, cache_len, attention_dim)
    
    def _get_mla_attention_dim(self, num_attention_heads: int, qk_rope_head_dim: int,
                             qk_nope_head_dim: int, kv_lora_rank: int) -> int:
        """Get attention dimension for MLA - to be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")


class PrefillFlopCounter(FlopCounter):
    
    @property
    def metric_name(self) -> str:
        return "Prefill-FLOP(T)"
    
    def _get_seq_len_for_phase(self, **kwargs) -> int:
        """For prefill, use the full sequence length."""
        return kwargs.get('seq_len', DEFAULT_SEQ_LEN)
    
    def _mla_kv_proj(self, seq_len: int, kv_lora_rank: int, num_attention_heads: int,
                    head_dim: int, qk_nope_head_dim: int) -> float:
        """Calculate FLOPs for MLA KV projection in prefill phase."""
        return self.gemm_flops(seq_len, kv_lora_rank, 
                             num_attention_heads * (head_dim + qk_nope_head_dim))
    
    def _get_mla_attention_dim(self, num_attention_heads: int, qk_rope_head_dim: int,
                             qk_nope_head_dim: int, kv_lora_rank: int) -> int:
        """Get attention dimension for MLA in prefill phase."""
        return num_attention_heads * (2 * qk_nope_head_dim + qk_rope_head_dim)

class DecodeFlopCounter(FlopCounter):
    
    @property
    def metric_name(self) -> str:
        return "Decode-FLOP(T)"
    
    def _get_seq_len_for_phase(self, **kwargs) -> int:
        """For decode, process only one token at a time."""
        return 1
    
    def _mla_kv_proj(self, seq_len: int, kv_lora_rank: int, num_attention_heads: int,
                    head_dim: int, qk_nope_head_dim: int) -> float:
        """Calculate FLOPs for MLA KV projection in decode phase."""
        flops = self.gemm_flops(seq_len, num_attention_heads, kv_lora_rank * qk_nope_head_dim)
        flops += self.gemm_flops(seq_len, num_attention_heads, kv_lora_rank * head_dim)
        return flops
    
    def _get_mla_attention_dim(self, num_attention_heads: int, qk_rope_head_dim: int,
                             qk_nope_head_dim: int, kv_lora_rank: int) -> int:
        """Get attention dimension for MLA in decode phase."""
        return num_attention_heads * (2 * kv_lora_rank + qk_rope_head_dim)
