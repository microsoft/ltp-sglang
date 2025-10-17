# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import pytest
import torch

from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.test.numerical_tests.comparison_module import CompareModule
from sglang.test.numerical_tests.utils.common import (
    BENCHMARK_FOLDER,
    LOG_DIR,
    TEST_DTYPES,
)


class TestTokenEmbeddingComparison(CompareModule):
    """Test class for comparing Token Embedding implementations with benchmarks."""

    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_token_embedding_comparison(self, dtype):
        """Run the Token Embedding comparison with traced tensors in the given benchmarks."""

        def module_init_func(module_config):
            """Initialize the Token Embedding module with the given configuration."""
            return VocabParallelEmbedding(
                module_config.vocab_size,
                module_config.hidden_size,
            )

        def module_forward_func(sgl_module, inputs, trace_metadata):
            """Forward function for the Token Embedding module."""
            input_tensor = inputs["input_ids"].cuda()
            # Forward the module
            output = sgl_module(input_tensor)
            return output

        self._test_module_comparison(
            module_init_func,
            module_forward_func,
            dtype=dtype,
            bench_dir=BENCHMARK_FOLDER,
            log_dir=os.path.join(LOG_DIR, "token_embedding-compare", f"{dtype}"),
        )
