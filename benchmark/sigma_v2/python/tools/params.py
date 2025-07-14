import json

def embedding_lmhead(config):
    """
    Calculate the embedding lmhead parameter count.
    """
    vocab_size = config["vocab_size"]
    hidden_size = config.get("hidden_size", config.get("n_embd", 768))
    return vocab_size * hidden_size * 2

def mha_attention(config):
    """
    Calculate the multi-head attention parameter count.
    """
    num_attention_heads = config.get("num_attention_heads", config.get("n_head", 12))
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    hidden_size = config.get("hidden_size", config.get("n_embd", 768))
    head_dim = config.get("head_dim", hidden_size // num_attention_heads)

    qkv = hidden_size * (num_attention_heads * head_dim +
                         num_key_value_heads * head_dim * 2)
    o = hidden_size * (num_attention_heads * head_dim)
    return qkv + o

def mla_attention(config):
    """
    Calculate the multi-layer attention parameter count using LoRA.
    """
    num_attention_heads = config["num_attention_heads"]
    hidden_size = config["hidden_size"]
    q_lora_rank = config.get("q_lora_rank", 0) if config.get("q_lora_rank") is not None else 0
    kv_lora_rank = config.get("kv_lora_rank", 0)
    qk_rope_head_dim = config.get("qk_rope_head_dim", 0)
    qk_nope_head_dim = config.get("qk_nope_head_dim", 0)
    v_head_dim = config.get("v_head_dim", 128)
    qkv_proj_lora = (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * hidden_size
    qb_proj_lora = q_lora_rank * num_attention_heads * (qk_rope_head_dim + qk_nope_head_dim)
    kv_b_proj_lora = kv_lora_rank * num_attention_heads * (v_head_dim + qk_nope_head_dim)
    o_proj = hidden_size * (num_attention_heads * v_head_dim)
    
    return qkv_proj_lora + qb_proj_lora + kv_b_proj_lora + o_proj

def mlp(config):
    """
    Calculate the MLP parameter count.
    """
    hidden_size = config.get("hidden_size", config.get("n_embd", 768))
    intermediate_size = config.get("intermediate_size", 4 * hidden_size)
    # Two linear layers: hidden -> intermediate and intermediate -> hidden.
    return hidden_size * intermediate_size + intermediate_size * hidden_size

def gateup_mlp(config):
    """
    Calculate the gated MLP parameter count.
    """
    hidden_size = config.get("hidden_size", config.get("n_embd", 768))
    if "intermediate_size" in config:
        intermediate_size = config["intermediate_size"]
    elif "n_inner" in config:
        intermediate_size = config["n_inner"]
    else:
        intermediate_size = 4 * hidden_size
    # Two computations per layer.
    return hidden_size * intermediate_size * 2 + intermediate_size * hidden_size

def moe(config):
    """
    Calculate the MoE parameter count.
    """
    hidden_size = config.get("hidden_size", config.get("n_embd", 768))
    if "n_routed_experts" in config:
        num_experts = config["n_routed_experts"] + config.get("n_shared_experts", 0)
    elif "num_local_experts" in config:
        num_experts = config["num_local_experts"]
    elif "num_experts" in config:
        num_experts = config["num_experts"]
    else:
        raise ValueError("No expert configuration found.")
    
    intermediate_size = config.get("moe_intermediate_size", config.get("intermediate_size", 4 * hidden_size))
    return num_experts * (hidden_size * intermediate_size * 2 +
                          intermediate_size * hidden_size + hidden_size)
    
def get_params(model_config):
    """
    Calculate the parameter counts for each model based on its configuration file.
    
    model_config: dict mapping model names to configuration file paths.
    """
    model_params = {}
    for model, config_path in model_config.items():
        with open(config_path, "r") as file:
            config = json.load(file)

        layers = config.get("num_hidden_layers", config.get("n_layer", 0))
        count = embedding_lmhead(config)

        if model == "dpsk":
            count += mla_attention(config) * layers
        else:
            count += mha_attention(config) * layers

        if model in ["qwen", "qwen_sigma"]:
            count += moe(config) * layers
        elif model == "dpsk":
            count += gateup_mlp(config) * config["first_k_dense_replace"]
            count += moe(config) * (layers - config["first_k_dense_replace"])

        model_params[model] = count

    return model_params


def get_param_size(model_config):
    """
    Calculate the parameter counts for each model based on its configuration file.
    
    model_config: dict mapping model names to configuration file paths.
    """
    model_params = {}
    for model, config_path in model_config.items():
        with open(config_path, "r") as file:
            config = json.load(file)

        layers = config.get("num_hidden_layers", config.get("n_layer", 0))
        count = embedding_lmhead(config) * 2

        if model == "dpsk":
            count += mla_attention(config) * layers * 2
        else:
            count += mha_attention(config) * layers * 2

        if model in ["qwen", "qwen_sigma"]:
            count += moe(config) * layers * 2
        elif model == "dpsk":
            count += gateup_mlp(config) * config["first_k_dense_replace"]
            count += moe(config) * (layers - config["first_k_dense_replace"])

        model_params[model] = count

    return model_params