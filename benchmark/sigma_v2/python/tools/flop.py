import json
from typing import Dict, Any
from util import get_config_value

def embedding_lmhead(config: Dict[str, Any], seq_len: int = 1024) -> int:
    """Calculate FLOPs for embedding and language model head."""
    vocab_size = config["vocab_size"]
    hidden_size = get_config_value(config, ["hidden_size", "n_embd"], 768)
    return 2 * vocab_size * hidden_size * seq_len

def mha_attention(config: Dict[str, Any], seq_len: int = 1024, kv_cache: int = 0) -> int:
    """Calculate FLOPs for multi-head attention."""
    num_attention_heads = get_config_value(config, ["num_attention_heads", "n_head"], 12)
    num_key_value_heads = get_config_value(config, ["num_key_value_heads"], num_attention_heads)
    hidden_size = get_config_value(config, ["hidden_size", "n_embd"], 768)
    head_dim = get_config_value(config, ["head_dim"], hidden_size // num_attention_heads)

    qkv_proj = hidden_size * (num_attention_heads * head_dim + 
                             num_key_value_heads * head_dim * 2) * 2 * seq_len
    o_proj = hidden_size * (num_attention_heads * head_dim) * 2 * seq_len
    
    if kv_cache == 0:
        attn = 2 * seq_len * seq_len * num_attention_heads * head_dim
    else:
        attn = 2 * seq_len * kv_cache * num_attention_heads * head_dim
        
    return qkv_proj + o_proj + attn

def mla_attention(config: Dict[str, Any], seq_len: int = 1024, kv_cache: int = 0) -> int:
    """Calculate FLOPs for multi-level attention."""
    num_attention_heads = config["num_attention_heads"]
    hidden_size = config["hidden_size"]
    q_lora_rank = get_config_value(config, ["q_lora_rank"], 0)
    kv_lora_rank = get_config_value(config, ["kv_lora_rank"], 0)
    qk_rope_head_dim = get_config_value(config, ["qk_rope_head_dim"], 0)
    qk_nope_head_dim = get_config_value(config, ["qk_nope_head_dim"], 0)
    v_head_dim = get_config_value(config, ["v_head_dim"], 128)
    
    qkv_proj_lora = (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * hidden_size * 2 * seq_len
    qb_proj_lora = q_lora_rank * num_attention_heads * (qk_rope_head_dim + qk_nope_head_dim) * 2 * seq_len
    o_proj = hidden_size * (num_attention_heads * v_head_dim) * 2 * seq_len

    if kv_cache == 0:
        kv_b_proj_lora = kv_lora_rank * num_attention_heads * (v_head_dim + qk_nope_head_dim) * 2 * seq_len
        attn = 2 * num_attention_heads * seq_len * seq_len * (2 * qk_nope_head_dim + qk_rope_head_dim)
        return qkv_proj_lora + qb_proj_lora + kv_b_proj_lora + o_proj + attn
    else:
        q_nope_out = seq_len * num_attention_heads * kv_lora_rank * qk_nope_head_dim * 2
        attn_bmm = seq_len * num_attention_heads * kv_lora_rank * qk_nope_head_dim * 2
        attn = 2 * num_attention_heads * seq_len * kv_cache * (2 * kv_lora_rank + qk_rope_head_dim)
        return qkv_proj_lora + qb_proj_lora + q_nope_out + attn_bmm + o_proj + attn

def gateup_mlp(config: Dict[str, Any], seq_len: int = 1024) -> int:
    """Calculate FLOPs for gated MLP."""
    hidden_size = get_config_value(config, ["hidden_size", "n_embd"])
    intermediate_size = get_config_value(config, ["intermediate_size", "n_inner"], 4 * hidden_size)
    return 2 * seq_len * hidden_size * intermediate_size * 3 + seq_len * intermediate_size

def moe(config: Dict[str, Any], seq_len: int = 1024) -> int:
    """Calculate FLOPs for Mixture of Experts."""
    hidden_size = get_config_value(config, ["hidden_size", "n_embd"])
    num_experts_per_tok = config["num_experts_per_tok"] + get_config_value(config, ["n_shared_experts"], 0)
    num_experts = get_config_value(config, ["n_routed_experts", "num_local_experts", "num_experts"])
    intermediate_size = get_config_value(config, ["moe_intermediate_size"], config["intermediate_size"])
    
    return (2 * seq_len * num_experts_per_tok * 
            (hidden_size * intermediate_size * 3) + 
            2 * seq_len * hidden_size * num_experts)

def get_model_flops(model_name: str, config_path: str, seq_len: int = 1, kv_cache: int = 0) -> int:
    """Calculate total FLOPs for a specific model architecture.
    
    Args:
        model_name: Name of the model architecture
        config_path: Path to the model configuration JSON file
        seq_len: Sequence length
        kv_cache: KV cache size (0 for no cache)
        
    Returns:
        Total number of FLOPs
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        raise ValueError(f"Error loading config from {config_path}: {e}")
        
    total_layers = get_config_value(config, ["num_hidden_layers", "n_layer"], 0)
    dense_layers = get_config_value(config, ["first_k_dense_replace"], 0)    
    
    count = embedding_lmhead(config, seq_len=seq_len)
    
    if model_name == "dpsk":
        count += mla_attention(config, seq_len=seq_len, kv_cache=kv_cache) * total_layers
    else:
        count += mha_attention(config, seq_len=seq_len, kv_cache=kv_cache) * total_layers
    count += gateup_mlp(config, seq_len=seq_len) * dense_layers
    count += moe(config, seq_len=seq_len) * (total_layers - dense_layers)    
    return count

def get_flops(model_config: Dict[str, str], seq_len: int = 1024, kv_cache: int = 0) -> Dict[str, int]:
    """Calculate FLOPs for multiple models.
    
    Args:
        model_config: Dictionary mapping model names to their config paths
        seq_len: Sequence length
        kv_cache: KV cache size (0 for no cache)
        
    Returns:
        Dictionary mapping model names to their FLOPs count
    """
    model_flops = {}
    for model, config_path in model_config.items():
        try:
            model_flops[model] = get_model_flops(model, config_path, seq_len=seq_len, kv_cache=kv_cache)
        except Exception as e:
            print(f"Error calculating FLOPs for {model}: {e}")
            model_flops[model] = None
    return model_flops