import sys
import uuid
from operator import pos

import pytest
import torch
from torch import nn

from sglang.srt.layers.rotary_embedding import get_rope
from sglang.test.numerical_tests.common import *
from sglang.test.numerical_tests.test_module import TestModule
from sglang.test.numerical_tests.utils.load_data import (
    load_random_weights,
    load_weight_from_hf_ckp,
)
from sglang.test.numerical_tests.utils.tensor_checker import *

# Define the configurations for the Rotary Embedding tests
ROPE_configs = [
    {
        "head_dim": 128,
        "num_attention_heads": 64,
        "max_position_embeddings": 4096,
        "rope_theta": 10000,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
    }
]


class TestRoPE(TestModule):
    """
    Test the consistency of Rotary Embedding Layer computation
    """

    @pytest.mark.parametrize("rope_config", ROPE_configs)
    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_sglang_rope(self, rope_config: dict, dtype: torch.dtype):
        """Test the Rotary Embedding layer."""
        # Initialize the Rotary Embedding module
        rope_module = get_rope(
            head_size=rope_config["head_dim"],
            rotary_dim=rope_config["head_dim"],
            max_position=rope_config["max_position_embeddings"],
            base=rope_config["rope_theta"],
            rope_scaling=rope_config["rope_scaling"],
        )
        rope_module = rope_module.to(dtype=dtype).cuda()
        rope_module.eval()

        print(f"Testing RoPE with weights: {rope_config=} " f" {dtype=}")

        log_dir = os.path.join(
            LOG_DIR,
            "rope",
            f"{rope_config['head_dim']}_{rope_config['num_attention_heads']}",
            f"rope-{uuid.uuid4().hex[:8]}",
        )
        os.makedirs(log_dir, exist_ok=True)

        def forward_func(inputs) -> torch.Tensor:
            """Forward function for the RoPE module."""
            positions, q, k = inputs
            # Clone the input tensor for avoiding in-place operations
            q_clone = q.clone()
            k_clone = k.clone()
            q_res, v_res = rope_module(positions, q_clone, k_clone)
            # merge the results
            return torch.cat((q_res, v_res), dim=-1)

        def random_input_method(bs, sl, dtype):
            """Generate random input tensor for RoPE."""
            positions = torch.arange(sl, dtype=torch.int64).cuda()
            q = torch.randn(
                (sl, rope_config["head_dim"], rope_config["num_attention_heads"]),
                dtype=dtype,
            ).cuda()
            k = torch.randn(
                (sl, rope_config["head_dim"], rope_config["num_attention_heads"]),
                dtype=dtype,
            ).cuda()
            return positions, q, k

        self._run_module_random_input(forward_func, random_input_method, dtype, log_dir)
