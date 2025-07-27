import sys
import uuid

import pytest
import torch
from torch import nn

from sglang.test.numerical_tests.common import *
from sglang.test.numerical_tests.modules.test_attention import AttentionLayer
from sglang.test.numerical_tests.test_module import TestModule
from sglang.test.numerical_tests.utils.load_data import (
    load_random_weights,
    load_weight_from_hf_ckp,
)
from sglang.test.numerical_tests.utils.tensor_checker import *

# Define the configurations for the Attention Layer tests
AttentionLayer_configs = [
    {
        "hidden_size": 5120,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "rope_theta": 10000,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
        "max_position_embeddings": 4096,
        "head_dim": 128,
        "rms_norm_eps": 1e-06,
        "attention_bias": False,
        "qk_layernorm": True,
    }
]
# Real weight prefixes in the model checkpoint for the Attention Layer tests
real_weight_prefixs = [
    "model.layers.0.self_attn",
    "model.layers.20.self_attn",
    "model.layers.40.self_attn",
    "model.layers.62.self_attn",
]
random_weights = [0] * len(real_weight_prefixs)


class TestAttentionLayer(TestModule):
    """
    Test the consistency of Attention Layer computation.
    """

    @pytest.mark.parametrize("module_config", AttentionLayer_configs)
    @pytest.mark.parametrize(
        "random_weight", random_weights
    )  # Placeholder for random weights
    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_sglang_attention_layer(self, module_config, random_weight, dtype):
        """Test the Attention Layer with random weights."""
        # Create the Attention Layer module with random weights
        sgl_module = AttentionLayer(
            hidden_size=module_config["hidden_size"],
            num_heads=module_config["num_attention_heads"],
            num_kv_heads=module_config["num_key_value_heads"],
            rope_theta=module_config["rope_theta"],
            rope_scaling=module_config["rope_scaling"],
            max_position_embeddings=module_config["max_position_embeddings"],
            head_dim=module_config["head_dim"],
            rms_norm_eps=module_config["rms_norm_eps"],
            attention_bias=module_config["attention_bias"],
            qk_layernorm=module_config["qk_layernorm"],
        )
        load_random_weights(sgl_module, dtype=dtype)

        print(f"Testing sglang AttentionLayer with random weights: {dtype=}")
        log_dir = os.path.join(
            LOG_DIR,
            "attention_layer",
            f"{module_config['hidden_size']}-{module_config['num_attention_heads']}-{module_config['num_key_value_heads']}-{dtype}",
            f"weights-{weight_prefix}",
        )
        os.makedirs(log_dir, exist_ok=True)

        self._run_module_random_input(
            sgl_module.forward, (module_config["hidden_size"],), dtype, log_dir=log_dir
        )
