# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import torch

from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.test.numerical_tests.test_module import TestModule
from sglang.test.numerical_tests.utils.common import *
from sglang.test.numerical_tests.utils.load_data import (
    load_random_weights,
    load_weight_from_hf_ckp,
)
from sglang.test.numerical_tests.utils.module_config import MODULE_CONFIGS

# Define the configurations for the Token Embedding tests
weight_prefixs = [
    "model.embed_tokens",
    "random0",
    "random1",
    "random2",
]
if not CHECKPOINT_PATH:
    weight_prefixes = [item for item in weight_prefixes if "random" in item]


class TestTokenEmbedding(TestModule):
    """
    Test the consistency of Token Embedding Layer computation
    """

    @pytest.mark.parametrize("embedding_config", MODULE_CONFIGS)
    @pytest.mark.parametrize("weight_prefix", weight_prefixs)
    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_sglang_embedding(
        self, embedding_config: dict, weight_prefix: str, dtype: torch.dtype
    ):
        """Test the token embedding layer."""
        # Initialize the embedding module
        embedding_module = VocabParallelEmbedding(
            embedding_config["vocab_size"],
            embedding_config["hidden_size"],
        )
        if weight_prefix.count("random") > 0:
            # Load random weights for the embedding module
            embedding_module = load_random_weights(embedding_module, dtype=dtype)
        else:
            # Load real weights from a checkpoint for the embedding module
            embedding_module = load_weight_from_hf_ckp(
                embedding_module, weight_prefix, dtype=dtype
            )

        log_dir = os.path.join(
            LOG_DIR,
            "token_embedding",
            f"{embedding_config['hidden_size']}_{embedding_config['vocab_size']}",
            f"weights-{weight_prefix}",
        )
        os.makedirs(log_dir, exist_ok=True)

        def random_input_method(bs: int, sl: int, dtype: torch.dtype) -> torch.Tensor:
            """Generate a random input tensor for token embedding."""
            return torch.randint(
                low=0,
                high=embedding_config["vocab_size"],
                size=(bs, sl),
                dtype=torch.int64,
            ).cuda()

        self._run_module_random_input(
            embedding_module.forward,
            random_input_method,
            dtype,
            log_dir=log_dir,
        )
