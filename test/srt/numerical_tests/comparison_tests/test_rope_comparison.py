# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import pytest
import torch

from sglang.srt.layers.rotary_embedding import get_rope
from sglang.test.numerical_tests.comparison_module import CompareModule
from sglang.test.numerical_tests.utils.common import (
    BENCHMARK_FOLDER,
    LOG_DIR,
    TEST_DTYPES,
)


class TestRoPEComparison(CompareModule):
    """Test class for comparing RoPE implementations with benchmarks."""

    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_rope_comparison(self, dtype):
        """Test the RoPE implementation with traced tensors in the given benchmarks."""

        def module_init_func(module_config):
            """Initialize the RoPE module with the given configuration."""
            return get_rope(
                head_size=module_config.head_dim,
                rotary_dim=module_config.head_dim,
                max_position=module_config.max_position_embeddings,
                base=module_config.rope_theta,
                rope_scaling=module_config.rope_scaling,
            )

        def module_forward_func(sgl_module, inputs, trace_metadata):
            """Forward function for the RoPE module."""

            batch_size, num_heads, seq_len, head_size = inputs["query"].shape
            num_key_value_heads = inputs["key"].shape[1]
            # Reshape and permute traced tensors from Transformers to match SGLang's module
            query = (
                inputs["query"]
                .permute(0, 2, 1, 3)
                .reshape(batch_size * seq_len, num_heads * head_size)
                .cuda()
            )
            key = (
                inputs["key"]
                .permute(0, 2, 1, 3)
                .reshape(batch_size * seq_len, num_key_value_heads * head_size)
                .cuda()
            )
            position_ids = (
                torch.arange(0, seq_len, dtype=torch.int64, device=query.device)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )
            # Forward the module
            q_embed, k_embed = sgl_module(position_ids, query, key)

            # Reshape and permute the output tensors from SGLang's module to match Transformers' expectations
            q_embed = (
                q_embed.view(batch_size, seq_len, num_heads, head_size)
                .permute(0, 2, 1, 3)
                .cuda()
            )
            k_embed = (
                k_embed.view(batch_size, seq_len, num_key_value_heads, head_size)
                .permute(0, 2, 1, 3)
                .cuda()
            )
            # Concatenate the query and key embeddings for comparison
            res = torch.cat((q_embed, k_embed), dim=1)
            return res

        self._test_module_comparison(
            module_init_func,
            module_forward_func,
            dtype=dtype,
            bench_dir=BENCHMARK_FOLDER,
            log_dir=os.path.join(LOG_DIR, "rope-compare", f"{dtype}"),
        )
