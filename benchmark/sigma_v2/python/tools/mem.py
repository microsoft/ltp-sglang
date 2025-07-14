import json
from constant import model_precision, model_config, model_length
from collections import defaultdict
import numpy as np

SEQ_LEN = 1024
KV_CACHE = 0

def get_sum(ts):
    s = 0
    for t in ts.values():
        if isinstance(t, list):
            s += sum(t)
        else:
            s += t
    return s

def embedding_lmhead(config, bsz=1, seq_len=SEQ_LEN):
    seq_len *= bsz
    vocab_size = config["vocab_size"]
    hidden_size = config.get("hidden_size", config.get("n_embd", 768))
    ts = {
        "input": [seq_len * 8,  seq_len * hidden_size * 2],
        "param": [vocab_size * hidden_size * 2, vocab_size * hidden_size * 2],
        "output": [seq_len * hidden_size * 2],
    }
    return get_sum(ts), ts

def mha_attention(config, bsz=1, seq_len=SEQ_LEN, kv_cache=KV_CACHE, param_dtype=2):
    num_attention_heads = config.get("num_attention_heads", config.get("n_head", 12))
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    hidden_size = config.get("hidden_size", config.get("n_embd", 768))
    head_dim = config.get("head_dim", hidden_size // num_attention_heads)
    t = [
        seq_len * hidden_size * 2, # input
        hidden_size * (num_attention_heads * head_dim + num_key_value_heads * head_dim * 2) * param_dtype, # qkv_proj
        seq_len * (num_attention_heads * head_dim + num_key_value_heads * head_dim * 2), #qkv output
        seq_len * num_attention_heads * head_dim * 2, # q
        kv_cache * num_key_value_heads * head_dim * 2 * 2, # kv
        seq_len * (num_attention_heads * head_dim) * 2, # attn_output
        seq_len * (num_attention_heads * head_dim) * 2, # o_proj input
        hidden_size * (num_attention_heads * head_dim) * param_dtype, # o_proj
        seq_len * hidden_size * 2, # o_proj output
    ]
    ts = {
        "input": [bsz * seq_len * hidden_size * 2, #input
            bsz * seq_len * num_attention_heads * head_dim * 2, #q
            bsz * seq_len * (num_attention_heads * head_dim) * 2, #attn_output
        ],
        "kv_cache": [
            bsz * kv_cache * num_key_value_heads * head_dim * 2 * 2 # kv_cache
        ], 
        "param": [
            hidden_size * (num_attention_heads * head_dim + num_key_value_heads * head_dim * 2) * param_dtype, # qkv_proj
            hidden_size * (num_attention_heads * head_dim) * param_dtype, # o_proj
        ],
        "output": [
            bsz * seq_len * (num_attention_heads * head_dim + num_key_value_heads * head_dim * 2), # qkv output
            bsz * seq_len * (num_attention_heads * head_dim) * 2, # o_proj input
            bsz * seq_len * hidden_size * 2, # o_proj output
        ]
            
    }
    return get_sum(ts), ts

def mla_attention(config, bsz=1, seq_len=SEQ_LEN, kv_cache=KV_CACHE, param_dtype=2):
    num_attention_heads = config["num_attention_heads"]
    hidden_size = config["hidden_size"]
    q_lora_rank = config.get("q_lora_rank", 0) if config.get("q_lora_rank") is not None else 0
    kv_lora_rank = config.get("kv_lora_rank", 0)
    qk_rope_head_dim = config.get("qk_rope_head_dim", 0)
    qk_nope_head_dim = config.get("qk_nope_head_dim", 0)
    v_head_dim = config.get("v_head_dim", 128)
    qk_head_dim = qk_rope_head_dim + qk_nope_head_dim
    
    if kv_cache == 0:
        t = [
            seq_len * hidden_size * 2, # input
            (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * hidden_size * param_dtype, # fused_qkv_a_proj
            seq_len * (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * 2, # fused qkv a output
            seq_len * num_attention_heads * q_lora_rank * 2, # q_b_proj input
            q_lora_rank * num_attention_heads * qk_head_dim * param_dtype, # q_b_proj
            seq_len * num_attention_heads * qk_head_dim * 2, # q_b_proj output
            seq_len * kv_lora_rank * 2, # kv_b_proj input
            kv_lora_rank * num_attention_heads * (v_head_dim + qk_nope_head_dim) * param_dtype, # kv_b_proj
            seq_len * num_attention_heads * (v_head_dim + qk_nope_head_dim) * 2, # kv_b_proj output
            seq_len * num_attention_heads * qk_head_dim * 2, # q
            seq_len * num_attention_heads * qk_head_dim * 2, # k
            seq_len * num_attention_heads * qk_nope_head_dim * 2, # v
            seq_len * num_attention_heads * v_head_dim * 2, # attn output
            seq_len * num_attention_heads * v_head_dim * 2, # o_proj input
            num_attention_heads * v_head_dim * hidden_size * param_dtype, # o_proj
            seq_len * hidden_size * 2, # o_proj output
        ]
        ts = {
            "input": [
                bsz * seq_len * hidden_size * param_dtype, # input
                bsz * seq_len * num_attention_heads * q_lora_rank * param_dtype, # q_b_proj input
                bsz * seq_len * kv_lora_rank * param_dtype, # kv_b_proj input
                bsz * seq_len * num_attention_heads * qk_head_dim * 2, # q
                bsz * seq_len * num_attention_heads * qk_head_dim * 2, # k
                bsz * seq_len * num_attention_heads * qk_nope_head_dim * 2, # v
                bsz * seq_len * num_attention_heads * v_head_dim * param_dtype, # o_proj input
            ],
            "param": [
                (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * hidden_size * param_dtype, # fused_qkv_a_proj
                q_lora_rank * num_attention_heads * qk_head_dim * param_dtype, # q_b_proj
                kv_lora_rank * num_attention_heads * (v_head_dim + qk_nope_head_dim) * param_dtype, # kv_b_proj
                seq_len * num_attention_heads * v_head_dim * 2, # o_proj input
            ],
            "output": [
                bsz * seq_len * (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * 2, # fused qkv a output
                bsz * seq_len * num_attention_heads * qk_head_dim * 2, # q_b_proj output
                bsz * seq_len * num_attention_heads * (v_head_dim + qk_nope_head_dim) * 2, # kv_b_proj output
                bsz * seq_len * num_attention_heads * v_head_dim * 2, # attn output
                bsz * seq_len * hidden_size * 2, # o_proj output
            ]
        }
    else:
        t = [
            seq_len * hidden_size * 2, # input
            (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * hidden_size * param_dtype, # fused_qkv_a_proj
            seq_len * (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * 2, # fused qkv a output
            seq_len * q_lora_rank * 2, # q_b_proj input
            q_lora_rank * num_attention_heads * qk_head_dim * param_dtype, # q_b_proj
            seq_len * num_attention_heads * qk_head_dim * 2, # q_b_proj output
            seq_len * num_attention_heads * qk_nope_head_dim * 2, # q_nope
            num_attention_heads * kv_lora_rank * qk_nope_head_dim * param_dtype, # w_kc
            seq_len * num_attention_heads * kv_lora_rank * 2, # q_nope output
            seq_len * num_attention_heads * kv_lora_rank * 2, # q
            kv_cache * 1 * (kv_lora_rank + qk_rope_head_dim) * 2, # kv_cache
            seq_len * num_attention_heads * v_head_dim * 2, # attn output
            seq_len * num_attention_heads * v_head_dim * 2, # o_proj input
            num_attention_heads * v_head_dim * hidden_size * param_dtype, # o_proj
            seq_len * hidden_size * 2, # o_proj output
        ]
        ts = {
            "input": [
                bsz * seq_len * hidden_size * param_dtype, # input
                bsz * seq_len * q_lora_rank * param_dtype, # q_b_proj input
                bsz * seq_len * num_attention_heads * qk_nope_head_dim * 2, # q_nope
                bsz * seq_len * num_attention_heads * kv_lora_rank * 2, # q
                bsz * seq_len * num_attention_heads * v_head_dim * param_dtype, # o_proj input
            ],
            "kv_cache": [
                bsz * kv_cache * 1 * (kv_lora_rank + qk_rope_head_dim) * 2, # kv_cache
            ],
            "param": [
                (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * hidden_size * param_dtype, # fused_qkv_a_proj
                q_lora_rank * num_attention_heads * qk_head_dim * param_dtype, # q_b_proj
                num_attention_heads * kv_lora_rank * qk_nope_head_dim * param_dtype, # w_kc
                num_attention_heads * v_head_dim * hidden_size * param_dtype, # o_proj
            ],
            "output": [
                bsz * seq_len * (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * 2, # fused qkv a output
                bsz * seq_len * num_attention_heads * qk_head_dim * 2, # q_b_proj output
                bsz * seq_len * num_attention_heads * kv_lora_rank * 2, # q_nope output
                bsz * seq_len * num_attention_heads * v_head_dim * 2, # attn output
                bsz * seq_len * hidden_size * 2, # o_proj output
            ]
        }
    return get_sum(ts), ts

def mlp(config, bsz=1, seq_len=SEQ_LEN, param_dtype=2):
    seq_len *= bsz
    hidden_size = config.get("hidden_size", config.get("n_embd"))
    intermediate_size = config.get("intermediate_size", 4 * hidden_size)
    t = [
        seq_len * hidden_size * 2, # input
        hidden_size * intermediate_size * param_dtype, # dense1
        seq_len * intermediate_size * 2, # dense1 output
        seq_len * intermediate_size * 2, # dense2 input
        intermediate_size * hidden_size * param_dtype, # dense2
        seq_len * hidden_size * 2, # dense2 output
    ]
    ts = {
        "input": [seq_len * hidden_size * param_dtype, seq_len * intermediate_size * param_dtype],
        "param": [hidden_size * intermediate_size * param_dtype, intermediate_size * hidden_size * param_dtype],
        "output": [seq_len * intermediate_size * 2, seq_len * hidden_size * 2],
    }
    return get_sum(ts), ts

def gateup_mlp(config, bsz=1, seq_len=SEQ_LEN, param_dtype=2):
    seq_len *= bsz
    hidden_size = config.get("hidden_size", config.get("n_embd"))
    intermediate_size = config.get("intermediate_size",
                                   config.get("n_inner", 4 * hidden_size))
    t = [
        seq_len * hidden_size * 2, # input
        hidden_size * intermediate_size * 2 * param_dtype, # gateup_dense1
        seq_len * intermediate_size * 2 * 2, # gateup_dense1 output
        seq_len * intermediate_size * 2 * 2, # gateup_dense2 input
        intermediate_size * hidden_size * param_dtype,  # gateup_dense2
        seq_len * hidden_size * 2, # gateup_dense2 output
    ]
    ts = {
        "input": [seq_len * hidden_size * param_dtype, seq_len * intermediate_size * 2 * param_dtype],
        "param": [hidden_size * intermediate_size * 2 * param_dtype, intermediate_size * hidden_size * param_dtype],
        "output": [seq_len * intermediate_size * 2 * 2, seq_len * hidden_size * 2],
    }
    return get_sum(ts), ts

def moe(config, bsz=1, seq_len=SEQ_LEN, param_dtype=2):
    seq_len *= bsz
    hidden_size = config.get("hidden_size", config.get("n_embd"))
    num_experts_per_tok = config["num_experts_per_tok"]
    num_experts = config.get("n_routed_experts",
                             config.get("num_local_experts",
                                        config.get("num_experts")))
    if "n_routed_experts" in config:
        num_experts += config.get("n_shared_experts", 0)
    intermediate_size = config.get("moe_intermediate_size", config["intermediate_size"])
    t = [
        seq_len * hidden_size * 2, # input
        hidden_size * num_experts * 2, # moe_gate
        seq_len * num_experts * 2, # moe_gate output
        min(seq_len * num_experts_per_tok, num_experts) * hidden_size * intermediate_size * 2 * param_dtype, # moe_expert_dense1
        min(seq_len * num_experts_per_tok, num_experts) * intermediate_size * hidden_size * param_dtype, # moe_expert_dense2
        # num_experts * hidden_size * intermediate_size * 2 * param_dtype, # moe_expert_dense1
        # num_experts * intermediate_size * hidden_size * param_dtype, # moe_expert_dense2
        seq_len * num_experts_per_tok * hidden_size * 2, # moe_expert_output
    ]
    ts = {
        "input": [
            bsz * seq_len * hidden_size * 2, 
            bsz * seq_len * num_experts * 2
        ],
        "param": [
            hidden_size * num_experts * 2, 
            num_experts * hidden_size * intermediate_size * 2 * param_dtype, 
            num_experts * intermediate_size * hidden_size * param_dtype],
        "output": [
            bsz * seq_len * num_experts * 2, 
            bsz * seq_len * num_experts_per_tok * hidden_size * 2],
    }
    return sum(t), ts

def get_model_mem(model_name, config_path, bsz=1, seq_len=SEQ_LEN, kv_cache=KV_CACHE):
    param_dtype = model_precision.get(model_name, 2)  # Default to 2 for most models
    with open(config_path, "r") as f:
        config = json.load(f)
    
    layers = config.get("num_hidden_layers", config.get("n_layer"))
    counts = []
    mem_size, _ = embedding_lmhead(config, bsz=bsz, seq_len=seq_len)
    counts.append(mem_size)
    
    if model_name == "dpsk":
        mem_size, _ = mla_attention(config, bsz=bsz, seq_len=seq_len, kv_cache=kv_cache, param_dtype=2)
        counts.append(mem_size * layers)
    else:
        mem_size, _ = mha_attention(config, bsz=bsz, seq_len=seq_len, kv_cache=kv_cache, param_dtype=param_dtype)
        counts.append(mem_size * layers)

    if model_name in ["qwen", "qwen_sigma"]:
        mem_size, _ = moe(config, bsz=bsz, seq_len=seq_len, param_dtype=param_dtype)
        counts.append(mem_size * layers)
    elif model_name == "dpsk":
        mem_size, _ = gateup_mlp(config, bsz=bsz, seq_len=seq_len, param_dtype=param_dtype)
        counts.append(mem_size * config["first_k_dense_replace"])
        mem_size, _ = moe(config, bsz=bsz, seq_len=seq_len, param_dtype=param_dtype)
        counts.append(mem_size * (layers - config["first_k_dense_replace"]))
        
    return sum(counts)

def get_mem(model_config, bsz=1, seq_len=SEQ_LEN, kv_cache=KV_CACHE):
    model_memio = {}
    for model, config_path in model_config.items():
        model_memio[model] = get_model_mem(model, config_path, bsz=bsz, seq_len=seq_len, kv_cache=kv_cache)
    return model_memio

def get_model_mem_by_type(model_name, config_path, bsz=1, seq_len=SEQ_LEN, kv_cache=KV_CACHE):
    param_dtype = model_precision.get(model_name, 2)  # Default to 2 for most models
    with open(config_path, "r") as f:
        config = json.load(f)
    layers = config.get("num_hidden_layers", config.get("n_layer"))
    mem_summary = {
        "input": [],
        "kv_cache": [],
        "param": [],
        "output": []
    }
    _, mem_by_type = embedding_lmhead(config, bsz=bsz, seq_len=seq_len)
    for k, v in mem_by_type.items():
        mem_summary[k].extend(v)
   
    if model_name == "dpsk":
        _, mem_by_type = mla_attention(config, bsz=bsz, seq_len=seq_len, kv_cache=kv_cache, param_dtype=param_dtype)
    else:
        _, mem_by_type = mha_attention(config, bsz=bsz, seq_len=seq_len, kv_cache=kv_cache, param_dtype=param_dtype)
    for k, v in mem_by_type.items():
        mem_summary[k].extend(v * layers)

    if model_name in ["qwen", "qwen_sigma"]:
        _, mem_by_type = moe(config, bsz=bsz, seq_len=seq_len, param_dtype=param_dtype)
        for k, v in mem_by_type.items():
            mem_summary[k].extend(v * layers)
    elif model_name == "dpsk":
        _, mem_by_type = gateup_mlp(config, bsz=bsz, seq_len=seq_len, param_dtype=param_dtype)
        for k, v in mem_by_type.items():
            mem_summary[k].extend(v * config["first_k_dense_replace"])
        _, mem_by_type = moe(config, bsz=bsz, seq_len=seq_len, param_dtype=param_dtype)
        for k, v in mem_by_type.items():
            mem_summary[k].extend(v * (layers - config["first_k_dense_replace"]))
    per_type_mem_sum = {k: sum(v) for k, v in mem_summary.items()}
    return per_type_mem_sum
    
def get_mem_by_type(model_config, bsz=1, seq_len=SEQ_LEN, kv_cache=KV_CACHE):
    model_memio = {}
    for model, config_path in model_config.items():
        model_memio[model] = get_model_mem_by_type(model, config_path, bsz=bsz, seq_len=seq_len, kv_cache=kv_cache)
    return model_memio