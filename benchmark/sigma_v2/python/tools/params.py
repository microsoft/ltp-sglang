import json
from .util import get_config_value

def embedding_lmhead(config):
    """
    Calculate the embedding lmhead parameter count.
    """
    vocab_size = get_config_value(config, ["vocab_size"], 50257)
    hidden_size = get_config_value(config, ["hidden_size", "n_embd"], 768)
    return vocab_size * hidden_size * 2

def mha_attention(config):
    """
    Calculate the multi-head attention parameter count.
    """
    num_attention_heads = get_config_value(config, ["num_attention_heads", "n_head"], 12)
    num_key_value_heads = get_config_value(config, ["num_key_value_heads"], num_attention_heads)
    hidden_size = get_config_value(config, ["hidden_size", "n_embd"], 768)
    head_dim = get_config_value(config, ["head_dim"], hidden_size // num_attention_heads)

    qkv = hidden_size * (num_attention_heads * head_dim +
                         num_key_value_heads * head_dim * 2)
    o = hidden_size * (num_attention_heads * head_dim)
    return qkv + o

def mla_attention(config):
    """
    Calculate the multi-layer attention parameter count using LoRA.
    """
    num_attention_heads = get_config_value(config, ["num_attention_heads", "n_head"], 12)
    hidden_size = get_config_value(config, ["hidden_size", "n_embd"], 768)
    q_lora_rank = get_config_value(config, ["q_lora_rank"], 0)
    kv_lora_rank = get_config_value(config, ["kv_lora_rank"], 0)
    qk_rope_head_dim = get_config_value(config, ["qk_rope_head_dim"], 0)
    qk_nope_head_dim = get_config_value(config, ["qk_nope_head_dim"], 0)
    v_head_dim = get_config_value(config, ["v_head_dim"], 128)
    qkv_proj_lora = (q_lora_rank + kv_lora_rank + qk_rope_head_dim) * hidden_size
    qb_proj_lora = q_lora_rank * num_attention_heads * (qk_rope_head_dim + qk_nope_head_dim)
    kv_b_proj_lora = kv_lora_rank * num_attention_heads * (v_head_dim + qk_nope_head_dim)
    o_proj = hidden_size * (num_attention_heads * v_head_dim)
    
    return qkv_proj_lora + qb_proj_lora + kv_b_proj_lora + o_proj

def gateup_mlp(config):
    """
    Calculate the gated MLP parameter count.
    """
    hidden_size = get_config_value(config, ["hidden_size", "n_embd"], 768)
    intermediate_size = get_config_value(config, ["intermediate_size", "n_inner"], 4 * hidden_size)
    return hidden_size * intermediate_size * 2 + intermediate_size * hidden_size

def moe(config):
    """
    Calculate the MoE parameter count.
    """
    hidden_size = get_config_value(config, ["hidden_size", "n_embd"], 768)
    num_experts = get_config_value(config, ["num_experts", "num_local_experts", "n_routed_experts"], 1)
    num_experts += get_config_value(config, ["n_shared_experts"], 0)
    intermediate_size = get_config_value(config, ["moe_intermediate_size", "intermediate_size"], 4 * hidden_size)
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

        total_layers = get_config_value(config, ["num_hidden_layers", "n_layer"], 0)
        dense_layers = get_config_value(config, ["first_k_dense_replace"], 0)
        
        count = embedding_lmhead(config)

        if model == "dpsk":
            count += mla_attention(config) * total_layers
        else:
            count += mha_attention(config) * total_layers

        count += gateup_mlp(config) * dense_layers
        count += moe(config) * (total_layers - dense_layers)

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

        layers = get_config_value(config, ["num_hidden_layers", "n_layer"], 0)
        count = embedding_lmhead(config) * 2

        if model == "dpsk":
            count += mla_attention(config) * layers * 2
        else:
            count += mha_attention(config) * layers * 2

        if model in ["qwen", "qwen_sigma"]:
            count += moe(config) * layers * 2
        elif model == "dpsk":
            count += gateup_mlp(config) * get_config_value(config, ["first_k_dense_replace"], 0)
            count += moe(config) * (layers - get_config_value(config, ["first_k_dense_replace"], 0))

        model_params[model] = count

    return model_params