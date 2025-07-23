import json
import math
import os
import uuid

import torch
from safetensors import safe_open
from torch import nn

from sglang.test.comparison_test.common import *
from sglang.test.comparison_test.tensor_tracer import trace_tensors, tracing_enabled
from sglang.test.comparison_test.tf.load_weights import (
    load_random_weights,
    load_weight_from_hf_ckp,
)
from sglang.test.comparison_test.tf.test_moe import MoE

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
    }
]

# Define the real weight prefixes in the model checkpoint for the MoEGate tests
real_weight_prefixs = ["model.layers.0.mlp", "model.layers.62.mlp"]
# Define the random weight test count as the same as the real weight test count
random_weights = [0] * len(real_weight_prefixs)


@torch.inference_mode()
def _run_moe_random_input(moe: MoE, dtype: torch.dtype, log_dir: str):
    """Run the MoE with random input and trace tensors."""
    moe = moe.to(dtype=dtype).cuda()
    weight_file = os.path.join(
        log_dir,
    )
    # save the weights to a file
    with safe_open(weight_file, framework="pt", device="cpu", create=True) as f:
        for name, param in moe.named_parameters():
            f.set_tensor(name, param.data.cpu())

    with tracing_enabled(verbose=False) as tracer:
        for bs in BATCH_SIZES:
            for sl in SEQ_LENS:
                print(f"Testing MoE with random input: {bs=} {sl=}")
                # Create a random input tensor
                input_tensor = torch.randn(
                    bs, sl, moe.config["hidden_size"], dtype=dtype
                )
                for _ in range(REPEAT_COUNT):
                    print(f"    Repeat {_+1}/{REPEAT_COUNT}")
                    moe(input_tensor)
                # Save the traced tensors
                saved_path = os.path.join(log_dir, f"trace_random_input_{bs=}_{sl=}")
                tracer.save_traced_tensors(saved_path)
                print(f"Traced tensors saved to {saved_path}")


@pytest.mark.parametrize("config", MoE_configs)
@pytest.mark.parametrize("real_weight_prefix", real_weight_prefixs)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_moe_load_weights(config, real_weight_prefix, dtype):
    """Test MoE with real weights loaded from a checkpoint."""
    # Prepare the log directory to save the weight and traced tensors as benchmark
    log_dir = os.path.join(
        LOG_DIR,
        "tf",
        "moe",
        f"{config['hidden_size']}",
        f"real_weights_{real_weight_prefix}",
    )
    os.makedirs(log_dir, exist_ok=True)
    test_config = TestConfig(
        module_config=config,
        real_weight_prefix=real_weight_prefix,
        log_dir=log_dir,
        batch_sizes=BATCH_SIZES,
        seq_lens=SEQ_LENS,
        dtype=dtype,
        input_count=RANDOM_INPUT_COUNT,
        repeat_count=REPEAT_COUNT,
    )
    # Save the test configuration
    test_config.save_json(os.path.join(log_dir, TEST_CONFIG_FILE))

    # Initialize the MoE with the given configuration
    moe = MoE(config)
    load_weight_from_hf_ckp(moe, config, real_weight_prefix)

    # Run the MoE with random input and trace tensors
    _run_moe_random_input(moe, dtype, log_dir)


@pytest.mark.parametrize("config", MoE_configs)
@pytest.mark.parametrize("random_weight", random_weights)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_moe_random_input(config, random_weight, dtype):
    """Test MoE with random input."""
    # Prepare the log directory to save the weight and traced tensors as benchmark
    log_dir = os.path.join(
        LOG_DIR,
        "tf",
        "moe",
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
    # Save the test configuration
    test_config.save_json(os.path.join(log_dir, TEST_CONFIG_FILE))

    # Initialize the MoE with the given configuration, it do not need to load weights
    moe = MoE(config)
    load_random_weights(moe, random_weight)
