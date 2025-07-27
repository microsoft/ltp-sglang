import sys
import uuid

import pytest
import torch
from torch import nn

from sglang.srt.layers.moe.ep_moe.layer import EPMoE
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.test.numerical_tests.common import *
from sglang.test.numerical_tests.modules.test_moe import MoE
from sglang.test.numerical_tests.test_module import TestModule
from sglang.test.numerical_tests.utils.load_data import (
    load_random_weights,
    load_weight_from_hf_ckp,
)
from sglang.test.numerical_tests.utils.tensor_checker import *

# Define the configurations for the MoE tests
MoE_configs = [
    {
        "num_experts_per_tok": 8,
        "n_routed_experts": 96,
        "routed_scaling_factor": 1.0,
        "scoring_func": "sigmoid",
        "seq_aux": True,
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "hidden_size": 5120,
        "moe_intermediate_size": 2160,
        "n_shared_experts": None,
        "hidden_act": "silu",
    }
]
# MoE Implementation
MoE_modules = [
    FusedMoE,
    EPMoE,
]
# Define the real weight prefixes in the model checkpoint for the MoE tests
real_weight_prefixs = [
    "model.layers.0.mlp",
    "model.layers.20.mlp",
    "model.layers.40.mlp",
    "model.layers.50.mlp",
    "model.layers.62.mlp",
]
# Define the random weight test count as the same as the real weight test count
random_weights = [0] * len(real_weight_prefixs)


class TestMoE(TestModule):
    """
    Test the consistency of MoE Layer computation across different implementations.
    """

    def _forward_func(self, moe_module, input_tensor):
        return moe_module(input_tensor)

    @pytest.mark.parametrize("module_config", MoE_configs)
    @pytest.mark.parametrize("moe_module_impl", MoE_modules)
    @pytest.mark.parametrize("real_weights", real_weight_prefixs)
    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    @pytest.mark.parametrize("tp_size", TP_SIZES)
    def test_sglang_moe_real_weights(
        self, module_config, moe_module_impl, real_weights, dtype, tp_size
    ):
        """Run the sglang MoE with random input and trace tensors."""
        sgl_module = MoE(module_config, moe_module_impl)
        sgl_module = load_weight_from_hf_ckp(sgl_module, real_weights, dtype=dtype)

        print(
            f"Testing sglang MoE with real weights: {moe_module_impl.__class__} {real_weights=} {dtype=} {tp_size=}"
        )
        log_dir = os.path.join(
            LOG_DIR,
            "moe",
            f"{moe_module_impl.__class__.__name__}-{module_config['hidden_size']}-{dtype}-{tp_size}",
            f"real_weights-{real_weights}",
        )
        os.makedirs(log_dir, exist_ok=True)
        self._run_module_random_input(
            sgl_module, (module_config["hidden_size"],), dtype, log_dir=log_dir
        )

    @pytest.mark.parametrize("module_config", MoE_configs)
    @pytest.mark.parametrize("moe_module_impl", MoE_modules)
    @pytest.mark.parametrize("random_weight", random_weights)
    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    @pytest.mark.parametrize("tp_size", TP_SIZES)
    def test_sglang_moe_random_weights(
        self, module_config, moe_module_impl, random_weight, dtype, tp_size
    ):
        """Run the sglang MoE with random input and trace tensors."""
        sgl_module = MoE(module_config, moe_module_impl)
        sgl_module = load_random_weights(sgl_module, dtype=dtype)

        print(
            f"Testing sglang MoE with random weights: {moe_module_impl.__class__} {random_weight=} {dtype=} {tp_size=}"
        )
        log_dir = os.path.join(
            LOG_DIR,
            "moe",
            f"{moe_module_impl.__class__.__name__}-{module_config['hidden_size']}-{dtype}-{tp_size}",
            f"random_weights-{uuid.uuid4().hex[:8]}",
        )
        os.makedirs(log_dir, exist_ok=True)
        self._run_module_random_input(
            sgl_module, (module_config["hidden_size"],), dtype, log_dir=log_dir
        )
