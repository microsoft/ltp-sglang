# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.numerical_tests.modules.sglang.test_logits import (
    LogitsModule,
    MockLogitsConfig,
)
from sglang.test.numerical_tests.test_module import TestModule
from sglang.test.numerical_tests.utils.common import *
from sglang.test.numerical_tests.utils.load_data import (
    load_random_weights,
    load_weight_from_hf_ckp,
)
from sglang.test.numerical_tests.utils.module_config import MODULE_CONFIGS

weight_prefixes = ["", "random0", "random1", "random2", "random3"]


class TestLogits(TestModule):
    """
    Test the consistency of Logits Layer computation across different implementations.
    """

    @pytest.mark.parametrize("logits_config", MODULE_CONFIGS)
    @pytest.mark.parametrize("weight_prefix", weight_prefixes)
    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_sglang_logits(
        self, logits_config: dict, weight_prefix: str, dtype: torch.dtype
    ):
        """Test the Logits layer."""
        # Initialize the Logits module
        mock_config = MockLogitsConfig(
            vocab_size=logits_config["vocab_size"],
            hidden_size=logits_config["hidden_size"],
        )
        logits_module = LogitsModule(mock_config)
        if weight_prefix.count("random") > 0:
            # Load random weights
            load_random_weights(logits_module, dtype=dtype)
        else:
            # Load real weights from a model checkpoint
            load_weight_from_hf_ckp(logits_module, weight_prefix, dtype)

        log_dir = os.path.join(
            LOG_DIR,
            "logits",
            f"{mock_config.vocab_size}_{mock_config.hidden_size}",
            f"weights-{weight_prefix}",
        )
        os.makedirs(log_dir, exist_ok=True)

        def forward_func(inputs) -> torch.Tensor:
            """Forward function for the Logits module."""
            # Clone the input tensor to avoid in-place operations
            input_tensor, logits_metadata = inputs
            input_tensor_clone = input_tensor.clone()
            logits_res = logits_module(input_tensor_clone, logits_metadata)
            return logits_res.next_token_logits

        def random_input_func(bs, sl, dtype):
            tensor = torch.randn(bs * sl, mock_config.hidden_size, dtype=dtype).cuda()

            logit_metadata = LogitsMetadata(
                forward_mode=ForwardMode.EXTEND,
                extend_return_logprob=False,
                extend_seq_lens=torch.tensor([sl], dtype=torch.int64).cuda().repeat(bs),
            )
            return tensor, logit_metadata

        self._run_module_random_input(
            forward_func, random_input_func, dtype, log_dir=log_dir
        )
