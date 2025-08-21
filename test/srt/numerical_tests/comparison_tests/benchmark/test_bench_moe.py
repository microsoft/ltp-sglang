import os

import pytest
import torch

from sglang.test.numerical_tests.bench_module import BenchConfig, BenchModule
from sglang.test.numerical_tests.modules.transformers.test_moe import MoE
from sglang.test.numerical_tests.utils.common import *
from sglang.test.numerical_tests.utils.load_data import (
    load_random_weights,
    load_weight_from_hf_ckp,
)
from sglang.test.numerical_tests.utils.module_config import MODULE_CONFIGS

# Define the real weight prefixes in the model checkpoint for the MoE tests
weight_prefixes = [
    "model.layers.0.mlp",
    "model.layers.20.mlp",
    "model.layers.40.mlp",
    "model.layers.62.mlp",
    "random0",
    "random1",
    "random2",
    "random3",
]


class TestBenchMoE(BenchModule):
    """Test MoE Layer computation with a benchmark (Transformers' implementation)."""

    @pytest.mark.parametrize("module_config", MODULE_CONFIGS)
    @pytest.mark.parametrize("weight_prefix", weight_prefixes)
    def test_bench_moe(self, module_config, weight_prefix):
        """Run the sglang MoE with random input and trace tensors."""
        # Initialize the MoE module with the given configurations
        moe_module = MoE(module_config)
        if weight_prefix.count("random") > 0:
            load_random_weights(moe_module, TARGET_DTYPE)
        else:
            load_weight_from_hf_ckp(moe_module, weight_prefix, TARGET_DTYPE)
        print(f"Testing MoE with weights: {module_config=} {weight_prefix=} ")

        # Create a directory for tracing tensors
        log_dir = os.path.join(
            LOG_DIR,
            "bench",
            "moe",
            f"{module_config['hidden_size']}_{module_config['num_experts_per_tok']}_{module_config['n_routed_experts']}",
            f"weights-{weight_prefix}",
        )

        # Define the random input function for the MoE module
        def random_input_func(batch_size, seq_len, dtype=TARGET_DTYPE):
            """Generate random input tensor for the MoE module."""
            hidden_states = torch.randn(
                batch_size, seq_len, module_config["hidden_size"], dtype=dtype
            ).cuda()
            return {"hidden_states": hidden_states}

        # Define the forward function for the MoE module
        def forward_func(inputs):
            """Forward function for the MoE module."""
            hidden_states = inputs["hidden_states"]
            return moe_module(hidden_states)

        # Create a BenchConfig instance for the MoE module
        bench_config = BenchConfig(
            module_config=module_config,
            real_weight_prefix=weight_prefix,
            log_dir=log_dir,
            batch_sizes=BATCH_SIZES,
            seq_lens=SEQ_LENS,
            dtype=TARGET_DTYPE,
            input_count=RANDOM_INPUT_COUNT,
            repeat_count=REPEAT_COUNT,
        )

        # Run the MoE module with random input and trace tensors
        self._run_module_random_input(
            moe_module, bench_config, forward_func, random_input_func, log_dir
        )
