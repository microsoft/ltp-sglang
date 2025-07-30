import os

import pytest
from torch import nn

from sglang.test.numerical_tests.bench_module import BenchConfig, BenchModule
from sglang.test.numerical_tests.utils.common import *
from sglang.test.numerical_tests.utils.load_data import (
    load_random_weights,
    load_weight_from_hf_ckp,
)

# Define the configurations for Logits tests
Logits_configs = [
    {
        "vocab_size": 200064,
        "hidden_size": 5120,
    }
]

weight_prefixes = [
    "lm_head",
    "random0",
    "random1",
]


class TestBenchLogits(BenchModule):
    """
    Test the Logits Layer computation with a benchmark (Transformers' implementation).
    """

    @pytest.mark.parametrize("logits_config", Logits_configs)
    @pytest.mark.parametrize("weight_prefix", weight_prefixes)
    def test_bench_logits(self, logits_config, weight_prefix):
        """Run the sglang Logits with random input and trace tensors."""
        # Initialize the Logits module with the given configurations
        logits_module = nn.Linear(
            logits_config["hidden_size"], logits_config["vocab_size"]
        )
        logits_module.bias = None  # Ensure no bias term is used
        if weight_prefix.count("random") > 0:
            load_random_weights(logits_module, TARGET_DTYPE)
        else:
            load_weight_from_hf_ckp(logits_module, weight_prefix, TARGET_DTYPE)
        print(f"Testing Logits with weights: {logits_config=} {weight_prefix=} ")

        # Create a directory for tracing tensors
        log_dir = os.path.join(
            LOG_DIR,
            "bench",
            "logits",
            f"{logits_config['vocab_size']}_{logits_config['hidden_size']}",
            f"weights-{weight_prefix}",
        )

        # Define the random input function for the Logits layer
        def random_input_func(batch_size, seq_len, dtype=TARGET_DTYPE):
            """Generate random input tensor for the Logits layer."""
            hidden_states = torch.randn(
                batch_size, seq_len, logits_config["hidden_size"], dtype=dtype
            ).cuda()
            return {"hidden_states": hidden_states}

        # Define the forward function for the Logits layer
        def forward_func(inputs):
            """Forward function for the Logits layer."""
            hidden_states = inputs["hidden_states"]
            logits_output = logits_module(hidden_states)
            return logits_output

        # Create a BenchConfig instance for the Logits module
        bench_config = BenchConfig(
            module_config=logits_config,
            real_weight_prefix=weight_prefix,
            log_dir=log_dir,
            batch_sizes=BATCH_SIZES,
            seq_lens=SEQ_LENS,
            dtype=TARGET_DTYPE,
            input_count=RANDOM_INPUT_COUNT,
            repeat_count=REPEAT_COUNT,
        )

        # Run the Logits module with random input and trace tensors
        self._run_module_random_input(
            logits_module, bench_config, forward_func, random_input_func, log_dir
        )
