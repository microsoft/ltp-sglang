import os

import pytest
import torch

from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.numerical_tests.comparison_module import CompareModule
from sglang.test.numerical_tests.utils.common import (
    BENCHMARK_FOLDER,
    LOG_DIR,
    TEST_DTYPES,
)


class TestRMSNormComparison(CompareModule):
    """Test class for comparing RMSNorm implementations with benchmarks."""

    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_rmsnorm_comparison(self, dtype):
        """Run the RMSNorm comparison with traced tensors in the given benchmarks."""

        def module_init_func(module_config):
            """Initialize the RMSNorm module with the given configuration."""
            return RMSNorm(
                hidden_size=module_config["hidden_size"],
                eps=module_config["rms_norm_eps"],
            )

        def module_forward_func(sgl_module, inputs, trace_metadata):
            """Forward function for the RMSNorm module."""

            hidden_size = inputs["hidden_states"].shape[-1]
            # Reshape the input tensor for the SGLang module
            input_tensor = inputs["hidden_states"].cuda().view(-1, hidden_size)

            # Forward the module
            output = sgl_module(input_tensor)
            return output.view(
                trace_metadata.batch_size,
                trace_metadata.seq_len,
                hidden_size,
            )

        self._test_module_comparison(
            module_init_func,
            module_forward_func,
            dtype=dtype,
            bench_dir=BENCHMARK_FOLDER,
            log_dir=os.path.join(LOG_DIR, "rmsnorm-compare", f"{dtype}"),
        )
