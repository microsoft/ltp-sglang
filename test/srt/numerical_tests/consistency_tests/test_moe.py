# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import uuid

import pytest
import torch
from torch import nn

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.test.numerical_tests.modules.sglang.test_moe import MoE
from sglang.test.numerical_tests.test_module import TestModule
from sglang.test.numerical_tests.utils.common import *
from sglang.test.numerical_tests.utils.load_data import (
    load_random_weights,
    load_weight_from_hf_ckp,
)
from sglang.test.numerical_tests.utils.module_config import MODULE_CONFIGS

# MoE Implementation
MoE_modules = [
    FusedMoE,
]
# Define the real weight prefixes in the model checkpoint for the MoE tests
weight_prefixes = [
    "model.layers.0.mlp",
    "model.layers.20.mlp",
    "model.layers.40.mlp",
    "model.layers.62.mlp",
    "random0",
    "random1",
    "random2",
    "random3",
]


class TestMoE(TestModule):
    """
    Test the consistency of MoE Layer computation across different implementations.
    """

    @pytest.mark.parametrize("module_config", MODULE_CONFIGS)
    @pytest.mark.parametrize("moe_module_impl", MoE_modules)
    @pytest.mark.parametrize("weight_prefix", weight_prefixes)
    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_sglang_moe(self, module_config, moe_module_impl, weight_prefix, dtype):
        """Run the sglang MoE with random input and trace tensors."""
        sgl_module = MoE(module_config, moe_module_impl)
        if weight_prefix.count("random") > 0:
            sgl_module = load_random_weights(sgl_module, dtype=dtype)
        else:
            # Load real weights from the specified prefix
            sgl_module = load_weight_from_hf_ckp(sgl_module, weight_prefix, dtype=dtype)

        log_dir = os.path.join(
            LOG_DIR,
            "moe",
            f"{moe_module_impl.__name__}-{module_config['hidden_size']}-{dtype}",
            f"weights-{weight_prefix}",
        )
        os.makedirs(log_dir, exist_ok=True)

        def forward_func(input_tensor: torch.Tensor) -> torch.Tensor:
            """Forward function for the MoE module."""
            # Clone the input tensor for avoiding in-place operations
            input_tensor_clone = input_tensor.clone()
            return sgl_module(input_tensor_clone)

        def random_input_func(bs: int, sl: int, dtype: torch.dtype) -> torch.Tensor:
            """Generate a random input tensor for MoE."""
            return torch.randn(
                bs * sl, module_config["hidden_size"], dtype=dtype
            ).cuda()

        self._run_module_random_input(
            forward_func, random_input_func, dtype, log_dir=log_dir
        )
