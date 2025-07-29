from model_config import ModelConfig

def get_model_results(counter,
                     seq_len: int,
                     model_config: ModelConfig) -> int:
    """Calculate the total number of parameters in the model."""
    sum_count = 0
    
    # Embedding and LM head
    sum_count += counter.embedding_lmhead(model_config.vocab_size, model_config.hidden_size, seq_len=seq_len)
    
    # Attention layers
    sum_count += counter.attention(
        model_config.hidden_size,
        model_config.num_attention_heads,
        model_config.head_dim,
        model_config.num_key_value_heads,
        model_config.qk_rope_head_dim,
        model_config.qk_nope_head_dim,
        model_config.q_lora_rank,
        model_config.kv_lora_rank,
        seq_len=seq_len
    ) * model_config.total_num_layers

    # Dense MLP layers
    if model_config.total_dense_layers > 0:
        sum_count += counter.gateup_mlp(model_config.hidden_size, model_config.intermediate_size, seq_len=seq_len) * model_config.total_dense_layers

    # MoE layers
    if model_config.num_experts > 0:
        sum_count += counter.moe(
            model_config.hidden_size,
            model_config.moe_intermediate_size,
            model_config.num_experts,
            n_shared_experts=model_config.n_shared_experts,
            seq_len=seq_len
        ) * (model_config.total_num_layers - model_config.total_dense_layers)
    
    return sum_count
