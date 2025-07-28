import sys
import uuid

import pytest
import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.numerical_tests.common import *
from sglang.test.numerical_tests.test_module import TestModule
from sglang.test.numerical_tests.utils.load_data import (
    load_random_weights,
    load_weight_from_hf_ckp,
)
from sglang.test.numerical_tests.utils.tensor_checker import *

# Define the configurations for the RMSNorm tests
RMSNorm_configs = [{"hidden_size": 5120, "rms_norm_eps": 1e-6}]

weight_prefixes = [
    "model.layers.0.input_layernorm",
    "model.layers.20.post_attention_layernorm",
    "model.layers.40.self_attn.q_layernorm",
    "model.layers.60.self_attn.k_layernorm",
    "model.norm",
    "random0",
    "random1",
    "random2",
    "random3",
]


class TestRMSNorm(TestModule):
    """Test the consistency of RMSNorm Layer computation"""

    @pytest.mark.parametrize("module_config", RMSNorm_configs)
    @pytest.mark.parametrize("weight_prefix", weight_prefixes)
    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_sglang_rmsnorm(self, module_config, weight_prefix, dtype):
        """Test RMSNorm with random input."""
        # Init RMSNorm with the given configuration
        print(
            f"Testing RMSNorm with weights: {module_config=} {weight_prefix=} {dtype=}"
        )
        rmsnorm = RMSNorm(module_config["hidden_size"], module_config["rms_norm_eps"])
        if weight_prefix.count("random") == 0:
            # Load weights from a checkpoint if not a random prefix
            load_weight_from_hf_ckp(rmsnorm, weight_prefix, dtype)
        else:
            # Load random weights for the RMSNorm layer
            load_random_weights(rmsnorm, dtype)
        print(f"RMSNorm weights loaded: {rmsnorm}")
        log_dir = os.path.join(
            LOG_DIR,
            "rmsnorm",
            f"{module_config['hidden_size']}-{dtype}",
            f"weights-{weight_prefix}",
        )
        os.makedirs(log_dir, exist_ok=True)

        def forward_func(input_tensor: torch.Tensor) -> torch.Tensor:
            """Forward function for the RMSNorm module."""
            # Clone the input tensor for avoiding in-place operations
            input_tensor_clone = input_tensor.clone()
            return rmsnorm(input_tensor_clone)

        def random_input_func(bs, sl, dtype):
            """Generate random input tensor for RMSNorm."""
            return torch.randn(
                bs * sl, module_config["hidden_size"], dtype=dtype
            ).cuda()

        self._run_module_random_input(
            forward_func,
            random_input_func,
            dtype,
            log_dir,
        )
