import json

SEQ_LEN = 1024
KV_CACHE = 0

def embedding_lmhead(config, seq_len=SEQ_LEN):
    vocab_size = config["vocab_size"]
    hidden_size = config.get("hidden_size", config.get("n_embd", 768))
    return 2 * vocab_size * hidden_size * seq_len

def mha_attention(config, seq_len=SEQ_LEN, kv_cache=KV_CACHE):
    num_attention_heads = config.get("num_attention_heads", config.get("n_head", 12))
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    hidden_size = config.get("hidden_size", config.get("n_embd", 768))
    head_dim = config.get("head_dim", hidden_size // num_attention_heads)

    qkv_proj = hidden_size * (num_attention_heads * head_dim +
                              num_key_value_heads * head_dim * 2) * 2 * seq_len
    o_proj = hidden_size * (num_attention_heads * head_dim) * 2 * seq_len
    if kv_cache == 0:
        attn = 2 * seq_len * seq_len * num_attention_heads * head_dim # q x k x v
    else:
        attn = 2 * seq_len * kv_cache * num_attention_heads  * head_dim # q x k x v
    return qkv_proj + o_proj + attn

def mla_attention(config, seq_len=SEQ_LEN, kv_cache=KV_CACHE):
    num_attention_heads = config["num_attention_heads"]
    hidden_size = config["hidden_size"]
    q_lora_rank = config.get("q_lora_rank", 0) if config.get("q_lora_rank") is not None else 0
    kv_lora_rank = config.get("kv_lora_rank", 0)
    qk_rope_head_dim = config.get("qk_rope_head_dim", 0)
    qk_nope_head_dim = config.get("qk_nope_head_dim", 0)
    v_head_dim = config.get("v_head_dim", 128)
    qkv_proj_lora = (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * hidden_size * \
                    2 * seq_len
    qb_proj_lora = q_lora_rank * num_attention_heads * \
                    (qk_rope_head_dim + qk_nope_head_dim) * 2 * seq_len
    o_proj = hidden_size * (num_attention_heads * v_head_dim) * 2 * seq_len

    if kv_cache == 0:
        kv_b_proj_lora = kv_lora_rank * num_attention_heads * \
                      (v_head_dim + qk_nope_head_dim) * 2 * seq_len
        attn = 2 * num_attention_heads * seq_len * seq_len * (2 * qk_nope_head_dim + qk_rope_head_dim)
        return qkv_proj_lora + qb_proj_lora + kv_b_proj_lora + o_proj + attn
    else:
        q_nope_out = seq_len * num_attention_heads * kv_lora_rank * qk_nope_head_dim * 2
        attn_bmm = seq_len * num_attention_heads * kv_lora_rank * qk_nope_head_dim * 2
        attn = 2 * num_attention_heads * seq_len * kv_cache * (2 * kv_lora_rank + qk_rope_head_dim)
        return qkv_proj_lora + qb_proj_lora + q_nope_out + attn_bmm + o_proj + attn

def mla_attention_sigma(config, seq_len=SEQ_LEN, kv_cache=KV_CACHE):
    num_attention_heads = config["num_attention_heads"]
    hidden_size = config["hidden_size"]
    q_lora_rank = config.get("q_lora_rank", 0) if config.get("q_lora_rank") is not None else 0
    kv_lora_rank = config.get("kv_lora_rank", 0)
    qk_rope_head_dim = config.get("qk_rope_head_dim", 0)
    qk_nope_head_dim = config.get("qk_nope_head_dim", 0)
    v_head_dim = config.get("v_head_dim", 128)
    qkv_proj_lora = (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * hidden_size * \
                    2 * seq_len
    qb_proj_lora = q_lora_rank * num_attention_heads * \
                    (qk_rope_head_dim + qk_nope_head_dim) * 2 * seq_len
    kv_b_proj_lora = kv_lora_rank * num_attention_heads * \
                      (v_head_dim + qk_nope_head_dim) * 2 * seq_len
    o_proj = hidden_size * (num_attention_heads * v_head_dim) * 2 * seq_len

    if kv_cache == 0:
        attn = 2 * num_attention_heads * seq_len * seq_len * (2 * kv_lora_rank + qk_rope_head_dim)
    else:
        attn = 2 * num_attention_heads * seq_len * kv_cache * (2 * kv_lora_rank + qk_rope_head_dim)

    return qkv_proj_lora + qb_proj_lora + kv_b_proj_lora + o_proj + attn

def mlp(config, seq_len=SEQ_LEN):
    hidden_size = config.get("hidden_size", config.get("n_embd"))
    intermediate_size = config.get("intermediate_size", 4 * hidden_size)
    return 2 * seq_len * (hidden_size * intermediate_size +
                          intermediate_size * hidden_size) + seq_len * intermediate_size

def gateup_mlp(config, seq_len=SEQ_LEN):
    hidden_size = config.get("hidden_size", config.get("n_embd"))
    intermediate_size = config.get("intermediate_size",
                                   config.get("n_inner", 4 * hidden_size))
    return 2 * seq_len * hidden_size * intermediate_size * 3 + seq_len * intermediate_size

def moe(config, seq_len=SEQ_LEN):
    hidden_size = config.get("hidden_size", config.get("n_embd"))
    num_experts_per_tok = config["num_experts_per_tok"] + config.get("n_shared_experts", 0)
    num_experts = config.get("n_routed_experts",
                             config.get("num_local_experts",
                                        config.get("num_experts")))
    intermediate_size = config.get("moe_intermediate_size", config["intermediate_size"])
    return 2 * seq_len * num_experts_per_tok * (hidden_size * intermediate_size * 3 + intermediate_size // 2) + 2 * seq_len * hidden_size * num_experts

def get_model_flops(model_name, config_path, bsz=1, seq_len=SEQ_LEN, kv_cache=KV_CACHE):
    with open(config_path, "r") as f:
        config = json.load(f)
    layers = config.get("num_hidden_layers", config.get("n_layer"))
    count = embedding_lmhead(config, seq_len=seq_len)
    if model_name == "dpsk":
        count += mla_attention(config, seq_len=seq_len, kv_cache=kv_cache) * layers
    else:
        count += mha_attention(config, seq_len=seq_len, kv_cache=kv_cache) * layers

    if model_name in ["qwen", "qwen_sigma"]:
        count += moe(config, seq_len=seq_len) * layers
    elif model_name == "dpsk":
        count += gateup_mlp(config, seq_len=seq_len) * config["first_k_dense_replace"]
        count += moe(config, seq_len=seq_len) * (layers - config["first_k_dense_replace"])
    return count * bsz


def get_flops(model_config, bsz=1, seq_len=SEQ_LEN, kv_cache=KV_CACHE):
    model_flops = {}
    for model, config_path in model_config.items():
        model_flops[model] = get_model_flops(model, config_path, bsz=bsz, seq_len=seq_len, kv_cache=kv_cache)
    return model_flops