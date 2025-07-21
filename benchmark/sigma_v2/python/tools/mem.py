import json
from typing import Dict, Any, Tuple, List
from .constant import model_precision
from .util import get_config_value

SEQ_LEN: int = 1024
KV_CACHE: int = 0

def get_sum(ts) -> int:
    """
    Sums all values in the dictionary. If a value is a list, sum its elements.
    """
    s = 0
    for t in ts.values():
        if isinstance(t, list):
            s += sum(t)
        else:
            s += t
    return s

def embedding_lmhead(config: Dict[str, Any], seq_len: int = SEQ_LEN) -> Tuple[int, Dict[str, List[int]]]:
    """
    Calculates memory access for embedding and LM head layers.
    """
    vocab_size = get_config_value(config, ["vocab_size", "n_vocab"], 50257)
    hidden_size = get_config_value(config, ["hidden_size", "n_embd"], 768)
    
    memory_access = {
        "input": [
            seq_len * 8,                # Token IDs (int64)
            seq_len * hidden_size * 2   # Hidden states (float16)
        ],
        "param": [
            vocab_size * hidden_size * 2,    # Embedding weights
            vocab_size * hidden_size * 2     # LM head weights
        ],
        "output": [
            seq_len * hidden_size * 2   # Embedding output
        ],
    }
    return get_sum(memory_access), memory_access

def mha_attention(
    config: Dict[str, Any], 
    seq_len: int = SEQ_LEN, 
    kv_cache: int = KV_CACHE, 
    param_dtype: int = 2
) -> Tuple[int, Dict[str, List[int]]]:
    """
    Calculates memory access for multi-head attention layers.
    """
    num_attention_heads = get_config_value(config, ["num_attention_heads", "n_head"], 12)
    num_key_value_heads = get_config_value(config, ["num_key_value_heads"], num_attention_heads)
    hidden_size = get_config_value(config, ["hidden_size", "n_embd"], 768)
    head_dim = get_config_value(config, ["head_dim"], hidden_size // num_attention_heads)
    memory_access = {
        "input": [
            seq_len * hidden_size * 2, # input
            seq_len * num_attention_heads * head_dim * 2, # q
            seq_len * (num_attention_heads * head_dim) * 2, # attn_output
        ],
        "kv_cache": [
            kv_cache * num_key_value_heads * head_dim * 2 * 2 # kv_cache
        ], 
        "param": [
            hidden_size * (num_attention_heads * head_dim + num_key_value_heads * head_dim * 2) * param_dtype, # qkv_proj
            hidden_size * (num_attention_heads * head_dim) * param_dtype, # o_proj
        ],
        "output": [
            seq_len * (num_attention_heads * head_dim + num_key_value_heads * head_dim * 2), # qkv output
            seq_len * (num_attention_heads * head_dim) * 2, # o_proj input
            seq_len * hidden_size * 2, # o_proj output
        ]
    }
    return get_sum(memory_access), memory_access

def mla_attention(
    config: Dict[str, Any], 
    seq_len: int = SEQ_LEN, 
    kv_cache: int = KV_CACHE, 
    param_dtype: int = 2
) -> Tuple[int, Dict[str, List[int]]]:
    """
    Calculates memory access for multi-lora attention layers.
    """
    num_attention_heads = get_config_value(config, ["num_attention_heads", "n_head"], 12)
    hidden_size = get_config_value(config, ["hidden_size", "n_embd"], 768)
    q_lora_rank = get_config_value(config, ["q_lora_rank"], 0)
    kv_lora_rank = get_config_value(config, ["kv_lora_rank"], 0)
    qk_rope_head_dim = get_config_value(config, ["qk_rope_head_dim"], 0)
    qk_nope_head_dim = get_config_value(config, ["qk_nope_head_dim"], 0)
    v_head_dim = get_config_value(config, ["v_head_dim"], 128)
    qk_head_dim = qk_rope_head_dim + qk_nope_head_dim
    
    if kv_cache == 0:
        memory_access = {
            "input": [
                seq_len * hidden_size * param_dtype, # input
                seq_len * num_attention_heads * q_lora_rank * param_dtype, # q_b_proj input
                seq_len * kv_lora_rank * param_dtype, # kv_b_proj input
                seq_len * num_attention_heads * qk_head_dim * 2, # q
                seq_len * num_attention_heads * qk_head_dim * 2, # k
                seq_len * num_attention_heads * qk_nope_head_dim * 2, # v
                seq_len * num_attention_heads * v_head_dim * param_dtype, # o_proj input
            ],
            "param": [
                (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * hidden_size * param_dtype, # fused_qkv_a_proj
                q_lora_rank * num_attention_heads * qk_head_dim * param_dtype, # q_b_proj
                kv_lora_rank * num_attention_heads * (v_head_dim + qk_nope_head_dim) * param_dtype, # kv_b_proj
                seq_len * num_attention_heads * v_head_dim * 2, # o_proj input
            ],
            "output": [
                seq_len * (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * 2, # fused qkv a output
                seq_len * num_attention_heads * qk_head_dim * 2, # q_b_proj output
                seq_len * num_attention_heads * (v_head_dim + qk_nope_head_dim) * 2, # kv_b_proj output
                seq_len * num_attention_heads * v_head_dim * 2, # attn output
                seq_len * hidden_size * 2, # o_proj output
            ]
        }
    else:
        memory_access = {
            "input": [
                seq_len * hidden_size * param_dtype, # input
                seq_len * q_lora_rank * param_dtype, # q_b_proj input
                seq_len * num_attention_heads * qk_nope_head_dim * 2, # q_nope
                seq_len * num_attention_heads * kv_lora_rank * 2, # q
                seq_len * num_attention_heads * v_head_dim * param_dtype, # o_proj input
            ],
            "kv_cache": [
                kv_cache * 1 * (kv_lora_rank + qk_rope_head_dim) * 2, # kv_cache
            ],
            "param": [
                (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * hidden_size * param_dtype, # fused_qkv_a_proj
                q_lora_rank * num_attention_heads * qk_head_dim * param_dtype, # q_b_proj
                num_attention_heads * kv_lora_rank * qk_nope_head_dim * param_dtype, # w_kc
                num_attention_heads * v_head_dim * hidden_size * param_dtype, # o_proj
            ],
            "output": [
                seq_len * (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * 2, # fused qkv a output
                seq_len * num_attention_heads * qk_head_dim * 2, # q_b_proj output
                seq_len * num_attention_heads * kv_lora_rank * 2, # q_nope output
                seq_len * num_attention_heads * v_head_dim * 2, # attn output
                seq_len * hidden_size * 2, # o_proj output
            ]
        }
    return get_sum(memory_access), memory_access

def gateup_mlp(
    config: Dict[str, Any], 
    seq_len: int = SEQ_LEN, 
    param_dtype: int = 2
) -> Tuple[int, Dict[str, List[int]]]:
    """
    Calculates memory access for gated MLP layers.
    """
    hidden_size = get_config_value(config, ["hidden_size", "n_embd"], 768)
    intermediate_size = get_config_value(config, ["intermediate_size", "n_inner"], 4 * hidden_size)
    memory_access = {
        "input": [seq_len * hidden_size * param_dtype, seq_len * intermediate_size * 2 * param_dtype],
        "param": [hidden_size * intermediate_size * 2 * param_dtype, intermediate_size * hidden_size * param_dtype],
        "output": [seq_len * intermediate_size * 2 * 2, seq_len * hidden_size * 2],
    }
    return get_sum(memory_access), memory_access

def moe(
    config: Dict[str, Any], 
    seq_len: int = SEQ_LEN, 
    param_dtype: int = 2
) -> Tuple[int, Dict[str, List[int]]]:
    """
    Calculates memory access for Mixture-of-Experts layers.
    """
    hidden_size = get_config_value(config, ["hidden_size", "n_embd"], 768)
    num_experts_per_tok = config["num_experts_per_tok"]
    num_experts = get_config_value(config, ["n_routed_experts", "num_local_experts", "num_experts"], 1)
    num_experts += get_config_value(config, ["n_shared_experts"], 0)
    intermediate_size = get_config_value(config, ["moe_intermediate_size", "intermediate_size"], 4 * hidden_size)
    memory_access = {
        "input": [
            seq_len * hidden_size * 2, 
            seq_len * num_experts * 2
        ],
        "param": [
            hidden_size * num_experts * 2, 
            num_experts * hidden_size * intermediate_size * 2 * param_dtype, 
            num_experts * intermediate_size * hidden_size * param_dtype],
        "output": [
            seq_len * num_experts * 2, 
            seq_len * num_experts_per_tok * hidden_size * 2],
    }
    return get_sum(memory_access), memory_access

def get_model_mem(
    model_name: str, 
    config_path: str, 
    seq_len: int = SEQ_LEN, 
    kv_cache: int = KV_CACHE
) -> int:
    """
    Returns total memory usage for a model given its config.
    """
    param_dtype = model_precision.get(model_name, 2)  # Default to 2 for most models
    with open(config_path, "r") as f:
        config = json.load(f)
    
    total_layers = get_config_value(config, ["num_hidden_layers", "n_layer"], 0)
    dense_layers = get_config_value(config, ["first_k_dense_replace"], 0)    

    mem_size, _ = embedding_lmhead(config, seq_len=seq_len)
    counts = mem_size
    
    if model_name == "dpsk":
        mem_size, _ = mla_attention(config, seq_len=seq_len, kv_cache=kv_cache, param_dtype=2)
    else:
        mem_size, _ = mha_attention(config, seq_len=seq_len, kv_cache=kv_cache, param_dtype=param_dtype)
    counts += mem_size * total_layers

    mem_size, _ = gateup_mlp(config, seq_len=seq_len, param_dtype=param_dtype)
    counts += mem_size * dense_layers
    mem_size, _ = moe(config, seq_len=seq_len, param_dtype=param_dtype)
    counts += mem_size * (total_layers - dense_layers)

    return counts

def get_mem(
    model_config: Dict[str, str], 
    seq_len: int = SEQ_LEN, 
    kv_cache: int = KV_CACHE
) -> Dict[str, int]:
    """
    Returns memory usage for all models in the config dictionary.
    """
    model_memio: Dict[str, int] = {}
    for model, config_path in model_config.items():
        model_memio[model] = get_model_mem(model, config_path, seq_len=seq_len, kv_cache=kv_cache)
    return model_memio

def get_model_mem_by_type(
    model_name: str, 
    config_path: str, 
    seq_len: int = SEQ_LEN, 
    kv_cache: int = KV_CACHE
) -> Dict[str, int]:
    """
    Returns memory usage by type (input, kv_cache, param, output) for a model.
    """
    param_dtype = model_precision.get(model_name, 2)  # Default to 2 for most models
    with open(config_path, "r") as f:
        config = json.load(f)
    layers = config.get("num_hidden_layers", config.get("n_layer"))
    mem_summary: Dict[str, List[int]] = {
        "input": [],
        "kv_cache": [],
        "param": [],
        "output": []
    }
    _, mem_by_type = embedding_lmhead(config, seq_len=seq_len)
    for k, v in mem_by_type.items():
        mem_summary[k].extend(v)
   
    if model_name == "dpsk":
        _, mem_by_type = mla_attention(config, seq_len=seq_len, kv_cache=kv_cache, param_dtype=param_dtype)
    else:
        _, mem_by_type = mha_attention(config, seq_len=seq_len, kv_cache=kv_cache, param_dtype=param_dtype)
    for k, v in mem_by_type.items():
        mem_summary[k].extend(v * layers)

    if model_name in ["qwen", "qwen_sigma"]:
        _, mem_by_type = moe(config, seq_len=seq_len, param_dtype=param_dtype)
        for k, v in mem_by_type.items():
            mem_summary[k].extend(v * layers)
    elif model_name == "dpsk":
        _, mem_by_type = gateup_mlp(config, seq_len=seq_len, param_dtype=param_dtype)
        for k, v in mem_by_type.items():
            mem_summary[k].extend(v * get_config_value(config, "first_k_dense_replace", 0))
        _, mem_by_type = moe(config, seq_len=seq_len, param_dtype=param_dtype)
        for k, v in mem_by_type.items():
            mem_summary[k].extend(v * (layers - get_config_value(config, "first_k_dense_replace", 0)))
    per_type_mem_sum = {k: sum(v) for k, v in mem_summary.items()}
    return per_type_mem_sum
    
def get_mem_by_type(
    model_config: Dict[str, str], 
    seq_len: int = SEQ_LEN, 
    kv_cache: int = KV_CACHE
) -> Dict[str, Dict[str, int]]:
    """
    Returns memory usage by type for all models in the config dictionary.
    """
    model_memio: Dict[str, Dict[str, int]] = {}
    for model, config_path in model_config.items():
        model_memio[model] = get_model_mem_by_type(model, config_path, seq_len=seq_len, kv_cache=kv_cache)
    return model_memio