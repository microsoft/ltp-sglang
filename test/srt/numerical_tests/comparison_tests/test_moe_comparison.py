import os

import pytest

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.test.numerical_tests.comparison_module import CompareModule
from sglang.test.numerical_tests.modules.test_moe import MoE
from sglang.test.numerical_tests.utils.common import (
    BENCHMARK_FOLDER,
    LOG_DIR,
    TEST_DTYPES,
)

# MoE Implementation
MoE_modules = [
    FusedMoE,
]


class TestMoEComparison(CompareModule):
    """Test class for comparing MoE implementation with benchmarks."""

    @pytest.mark.parametrize("moe_impl", MoE_modules)
    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_moe_comparison(self, moe_impl, dtype):
        """Run the MoE comparison with traced tensors in the given benchmarks."""

        def module_init_func(module_config):
            """Initialize the MoE module with the given configuration."""
            return MoE(module_config, moe_impl)

        def module_forward_func(sgl_module, inputs, trace_metadata):
            """Forward function for the MoE module."""
            # Reshape the input tensor for the SGLang module
            input_tensor = (
                inputs["hidden_states"]
                .view(-1, sgl_module.config["hidden_size"])
                .cuda()
            )
            # Forward the module
            output = sgl_module(input_tensor)
            # Reshape it to the match the expected output shape
            return output.view(
                trace_metadata.batch_size,
                trace_metadata.seq_len,
                sgl_module.config["hidden_size"],
            )

        self._test_module_comparison(
            module_init_func,
            module_forward_func,
            dtype=dtype,
            bench_dir=BENCHMARK_FOLDER,
            log_dir=os.path.join(LOG_DIR, "moe-compare", f"{moe_impl.__name__}{dtype}"),
        )
