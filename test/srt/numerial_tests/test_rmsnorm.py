import sys

import pytest
import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.numerical_tests.common import *
from sglang.test.numerical_tests.test_module import TestModule
from sglang.test.numerical_tests.utils.load_data import (
    load_random_weights,
    load_weight_from_hf_ckp,
)
from sglang.test.numerical_tests.utils.tensor_checker import *

# Define the configurations for the RMSNorm tests
RMSNorm_configs = [{"hidden_size": 5120, "rms_norm_eps": 1e-6}]


class TestRMSNorm(TestModule):
    """Test the consistency of RMSNorm Layer computation"""

    @torch.inference_mode()
    def _run_rmsnorm_random_input(
        self, rmsnorm: RMSNorm, dtype: torch.dtype, log_dir: str
    ):
        """Run the RMSNorm with random input and trace tensors."""
        pass

    @pytest.mark.parametrize("config", RMSNorm_configs)
    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_rmsnorm_randomcd_input(self, config, dtype):
        """Test RMSNorm with random input."""
        rmsnorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])

        print(
            f"Testing RMSNorm with random input: {config['hidden_size']} {config['rms_norm_eps']}"
        )
        self._run_rmsnorm_random_input(rmsnorm, dtype, LOG_DIR)
