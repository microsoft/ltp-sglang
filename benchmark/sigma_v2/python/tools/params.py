
def embedding_lmhead(vocab_size: int, hidden_size: int, **kwargs) -> int:
    """Calculate the size of the embedding and LM head."""
    return vocab_size * hidden_size * 2 # embedding + lm_head

def gateup_mlp(hidden_size: int, intermediate_size: int, **kwargs) -> int:
    """Calculate the size of the gate-up MLP."""
    return hidden_size * intermediate_size * 2 + intermediate_size * hidden_size # up proj + down proj

def moe(hidden_size: int, moe_intermediate_size: int, num_experts: int, n_shared_experts: int = 0, **kwargs) -> int:
    """Calculate the size of the MoE layer."""
    return (
        hidden_size * num_experts # gate
        + (num_experts + n_shared_experts) * hidden_size * moe_intermediate_size * 2 # up proj
        + (num_experts + n_shared_experts) * hidden_size * moe_intermediate_size # down proj
    )

def attention(hidden_size: int, 
            num_attention_heads: int, 
            head_dim: int,
            num_key_value_heads: int, 
            qk_rope_head_dim: int, 
            qk_nope_head_dim: int, 
            q_lora_rank: int = 0, 
            kv_lora_rank: int = 0,
            **kwargs) -> int:
    """Calculate the size of the attention layer."""
    if qk_rope_head_dim == 0:
        if num_key_value_heads is None:
            qkv_proj = hidden_size * hidden_size * 3 # MHA
        else:
            qkv_proj = hidden_size * (num_attention_heads * head_dim + num_key_value_heads * head_dim * 2) # GQA
    else:
        qkv_proj = (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * hidden_size
        qkv_proj += q_lora_rank * num_attention_heads * (qk_rope_head_dim + qk_nope_head_dim)
        qkv_proj += kv_lora_rank * num_attention_heads * (head_dim + qk_nope_head_dim) # MLA with LoRA
    
    o_proj = hidden_size * (num_attention_heads * head_dim)
    
    return qkv_proj + o_proj 