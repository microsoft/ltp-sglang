# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import pytest
import torch

from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.test.numerical_tests.comparison_module import CompareModule
from sglang.test.numerical_tests.modules.sglang.test_attention import (
    AttentionLayer,
    AttentionLayerTester,
)
from sglang.test.numerical_tests.utils.common import (
    BENCHMARK_FOLDER,
    LOG_DIR,
    TEST_DTYPES,
)

attention_backends = [
    "triton",
    "torch_native",
]


class TestAttentionComparison(CompareModule):
    """Test class for comparing Attention implementation with benchmarks."""

    def setup_method(self, method):
        super().setup_method(method)
        # Initialize the distributed environment for Triton attention backend
        initialize_dp_attention(
            enable_dp_attention=False,
            tp_rank=0,
            tp_size=1,
            dp_size=1,
            moe_dense_tp_size=None,
            pp_size=1,
        )

    @pytest.mark.parametrize("attn_backend", attention_backends)
    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_attention_comparison(self, attn_backend, dtype):
        """Run the Attention comparison with  traced tensors in the given benchmarks."""

        def module_init_func(module_config):
            """Initialize the Attention module with the given configuration."""
            return AttentionLayer(
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

        def module_forward_func(sgl_module, inputs, trace_metadata):
            """Forward function for the Attention module."""
            module_tester = AttentionLayerTester(
                sgl_module,
                attn_backend=attn_backend,
                batch_size=trace_metadata.batch_size,
                seq_len=trace_metadata.seq_len,
            )
            # Reshape the input tensor for the SGLang module
            input_tensor = (
                inputs["hidden_states"].view(-1, sgl_module.hidden_size).cuda()
            )

            positions = (
                torch.arange(
                    trace_metadata.seq_len,
                    dtype=torch.int64,
                    device=input_tensor.device,
                )
                .repeat(trace_metadata.batch_size, 1)
                .unsqueeze(0)
            )
            # Forward the module
            output = module_tester.forward(positions, input_tensor)
            # Reshape it to the match the expected output shape
            return output.view(
                trace_metadata.batch_size,
                trace_metadata.seq_len,
                sgl_module.hidden_size,
            )

        self._test_module_comparison(
            module_init_func,
            module_forward_func,
            dtype=dtype,
            bench_dir=BENCHMARK_FOLDER,
            log_dir=os.path.join(
                LOG_DIR, "attention-compare", f"{attn_backend}-{dtype}"
            ),
        )
