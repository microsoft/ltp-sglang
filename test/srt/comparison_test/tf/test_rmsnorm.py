import json
import os
import uuid

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from sglang.test.comparison_test.common import *
from sglang.test.comparison_test.tensor_tracer import trace_tensors, tracing_enabled
from sglang.test.comparison_test.tf.test_rmsnorm import RMSNorm

# Define the configurations for the RMSNorm tests
RMSNorm_configs = [{"hidden_size": 5120, "rms_norm_eps": 1e-6}]


@torch.inference_mode()
def _run_rmsnorm_random_input(rmsnorm: RMSNorm, dtype: torch.dtype, log_dir: str):
    """Run the RMSNorm with random input and trace tensors."""
    rmsnorm = rmsnorm.to(dtype=dtype).cuda()
    weight_file = os.path.join(log_dir, WEIGHTS_FILE)
    # save the weights to a file
    with safe_open(weight_file, framework="pt", device="cpu", create=True) as f:
        f.set_tensor("weight", rmsnorm.weight.cpu())

    with tracing_enabled(verbose=False) as tracer:
        for bs in BATCH_SIZES:
            for sl in SEQ_LENS:
                print(f"Testing RMSNorm with random input: {bs=} {sl=}")
                # Create a random input tensor
                input_tensor = torch.randn(bs, sl, rmsnorm.weight.size(0), dtype=dtype)
                for _ in range(REPEAT_COUNT):
                    print(f"    Repeat {_+1}/{REPEAT_COUNT}")
                    rmsnorm(input_tensor)
                # Save the traced tensors
                saved_path = os.path.join(log_dir, f"trace_random_input_{bs=}_{sl=}")
                tracer.save_traced_tensors(saved_path)
                print(f"Traced tensors saved to {saved_path}")


@pytest.mark.parametrize("config", RMSNorm_configs)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_random_input(config, dtype):
    """Test RMSNorm with random input."""
    log_dir = os.path.join(
        LOG_DIR,
        "tf",
        "rmsnorm",
        f"{config['hidden_size']}",
        f"random_input_{uuid.uuid4()[:8]}",
    )
    os.makedirs(log_dir, exist_ok=True)
    test_config = TestConfig(
        module_config=config,
        log_dir=log_dir,
        batch_sizes=BATCH_SIZES,
        seq_lens=SEQ_LENS,
        dtype=dtype,
        input_count=RANDOM_INPUT_COUNT,
        repeat_count=REPEAT_COUNT,
    )
    test_config.save_json(os.path.join(log_dir, TEST_CONFIG_FILE))

    # Initialize the RMSNorm with the given configuration, it do not need to load weights
    rmsnorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])

    # Run the RMSNorm with random input and trace tensors
    print(f"Testing RMSNorm with random input: {config=} {dtype=}")
    _run_rmsnorm_random_input(rmsnorm, dtype, log_dir)
