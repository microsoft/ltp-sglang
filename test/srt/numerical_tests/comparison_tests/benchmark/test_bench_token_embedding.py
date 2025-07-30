import os

import pytest
import torch
from torch import nn

from sglang.test.numerical_tests.bench_module import BenchConfig, BenchModule
from sglang.test.numerical_tests.utils.common import *
from sglang.test.numerical_tests.utils.load_data import (
    load_random_weights,
    load_weight_from_hf_ckp,
)

# Define the configurations for the Token Embedding tests
Embedding_configs = [
    {
        "vocab_size": 200064,
        "hidden_size": 5120,
    }
]
weight_prefixes = [
    "model.embed_tokens",
    "random0",
    "random1",
    "random2",
]


class TestBenchTokenEmbedding(BenchModule):
    """
    Test the Token Embedding Layer computation with a benchmark (Transformers' implementation).
    """

    @pytest.mark.parametrize("embedding_config", Embedding_configs)
    @pytest.mark.parametrize("weight_prefix", weight_prefixs)
    def test_bench_embedding(self, embedding_config, weight_prefix):
        """Run the sglang Token Embedding with random input and trace tensors."""
        # Initialize the embedding module with the given configurations
        embedding_module = nn.Embedding(
            embedding_config["vocab_size"],
            embedding_config["hidden_size"],
        )
        if weight_prefix.count("random") > 0:
            load_random_weights(embedding_module, TARGET_DTYPE)
        else:
            load_weight_from_hf_ckp(embedding_module, weight_prefix, TARGET_DTYPE)
        print(
            f"Testing Token Embedding with weights: {embedding_config=} {weight_prefix=} "
        )

        # Create a directory for tracing tensors
        log_dir = os.path.join(
            LOG_DIR,
            "bench",
            "token_embedding",
            f"{embedding_config['hidden_size']}_{embedding_config['vocab_size']}",
            f"weights-{weight_prefix}",
        )

        # Define the random input function for the Token Embedding layer
        def random_input_func(batch_size, seq_len, dtype=TARGET_DTYPE):
            """Generate random input tensor for the Token Embedding layer."""
            input_ids = torch.randint(
                0,
                embedding_config["vocab_size"],
                (batch_size, seq_len),
                dtype=torch.int64,
            ).cuda()
            return {"input_ids": input_ids}

        # Define the forward function for the Token Embedding layer
        def forward_func(inputs):
            """Forward function for the Token Embedding layer."""
            input_ids = inputs["input_ids"]
            return embedding_module(input_ids)

        # Create a BenchConfig instance for the Token Embedding module
        bench_config = BenchConfig(
            module_config=embedding_config,
            real_weight_prefix=weight_prefix,
            log_dir=log_dir,
            batch_sizes=BATCH_SIZES,
            seq_lens=SEQ_LENS,
            dtype=TARGET_DTYPE,
            input_count=RANDOM_INPUT_COUNT,
            repeat_count=REPEAT_COUNT,
        )

        # Run the Token Embedding module with random input and trace tensors
        self._run_module_random_input(
            embedding_module, bench_config, forward_func, random_input_func, log_dir
        )
