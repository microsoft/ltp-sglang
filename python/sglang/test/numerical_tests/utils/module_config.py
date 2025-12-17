# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from transformers import PretrainedConfig

MODULE_CONFIGS = list(
    map(
        PretrainedConfig.from_dict,
        [
            {
                "attention_bias": False,
                "hidden_act": "silu",
                "hidden_size": 5120,
                "intermediate_size": 7168,
                "max_position_embeddings": 4096,
                "model_type": "sigma",
                "moe_intermediate_size": 2160,
                "n_routed_experts": 96,
                "n_shared_experts": 0,
                "norm_topk_prob": True,
                "head_dim": 128,
                "num_attention_heads": 64,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 63,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_scaling": None,
                "rope_theta": 10000,
                "routed_scaling_factor": 1.0,
                "scoring_func": "sigmoid",
                "seq_aux": True,
                "qk_layernorm": True,
                "tie_word_embeddings": False,
                "topk_method": "noaux_tc",
                "torch_dtype": "bfloat16",
                "transformers_version": "4.33.1",
                "vocab_size": 200064,
                "sliding_window": 4096,
                "_attn_implementation": "flash_attention_2",
            },
        ],
    )
)
