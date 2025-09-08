# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.numerical_tests.comparison_module import CompareModule
from sglang.test.numerical_tests.modules.sglang.test_logits import (
    LogitsModule,
    MockLogitsConfig,
)
from sglang.test.numerical_tests.utils.common import (
    BENCHMARK_FOLDER,
    LOG_DIR,
    TEST_DTYPES,
)


class TestLogitsComparison(CompareModule):
    """Test class for comparing Logits implementation with benchmarks."""

    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_logits_comparison(self, dtype):
        """Run the Logits comparison with traced tensors in the given benchmarks."""

        def module_init_func(module_config):
            """Initialize the Logits module with the given configuration."""
            vocab_size = module_config["vocab_size"]
            hidden_size = module_config["hidden_size"]
            return LogitsModule(MockLogitsConfig(vocab_size, hidden_size))

        def module_forward_func(sgl_module, inputs, trace_metadata):
            """Forward function for the Logits module."""

            hidden_size = inputs["hidden_states"].shape[-1]
            # Reshape the input tensor for the SGLang module
            input_tensor = inputs["hidden_states"].cuda().view(-1, hidden_size)
            logit_metadata = LogitsMetadata(
                forward_mode=ForwardMode.EXTEND,
                extend_return_logprob=False,
                extend_seq_lens=torch.tensor(
                    [trace_metadata.seq_len], dtype=torch.int64
                )
                .cuda()
                .repeat(trace_metadata.batch_size),
            )
            # Forward the module
            output = sgl_module(input_tensor, logit_metadata).next_token_logits.to(
                dtype
            )
            return output

        self._test_module_comparison(
            module_init_func,
            module_forward_func,
            dtype=dtype,
            bench_dir=BENCHMARK_FOLDER,
            log_dir=os.path.join(LOG_DIR, "logits-compare", f"{dtype}"),
            # Add a prefix for the module to match weights in the benchmark
            module_prefix="lm_head",
        )
