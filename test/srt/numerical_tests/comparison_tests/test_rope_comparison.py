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
                head_size=module_config["head_dim"],
                rotary_dim=module_config["head_dim"],
                max_position=module_config["max_position_embeddings"],
                base=module_config["rope_theta"],
                rope_scaling=module_config["rope_scaling"],
            )

        def module_forward_func(sgl_module, inputs, trace_metadata):
            """Forward function for the RoPE module."""
            # Clone and reshape the input tensors
            batch_size, num_heads, seq_len, head_size = inputs["query"].shape
            query = (
                inputs["query"]
                .clone()
                .permute(1, 0, 2, 3)
                .reshape(num_heads, batch_size * seq_len, head_size)
                .cuda()
            )
            key = (
                inputs["key"]
                .clone()
                .permute(1, 0, 2, 3)
                .reshape(num_heads, batch_size * seq_len, head_size)
                .cuda()
            )
            position_ids = torch.arange(
                0, seq_len, dtype=torch.int64, device=query.device
            ).unsqueeze(0)
            # Forward the module
            print(
                f"        RoPE input shapes: {query.shape}, {key.shape}, {position_ids.shape}"
            )
            q_embed, k_embed = sgl_module(position_ids, query, key)
            q_embed = (
                q_embed.view(num_heads, batch_size, seq_len, head_size)
                .permute(1, 2, 0, 3)
                .cuda()
            )
            k_embed = (
                k_embed.view(num_heads, batch_size, seq_len, head_size)
                .permute(1, 2, 0, 3)
                .cuda()
            )
            # Concatenate the query and key embeddings
            res = torch.cat((q_embed, k_embed), dim=-1)
            print(f"        RoPE output shape: {res.shape}")
            return res

        self._test_module_comparison(
            module_init_func,
            module_forward_func,
            dtype=dtype,
            bench_dir=BENCHMARK_FOLDER,
            log_dir=os.path.join(LOG_DIR, "rope-compare", f"{dtype}"),
        )
