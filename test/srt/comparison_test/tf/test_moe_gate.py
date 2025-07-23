import json
import math
import os
import uuid

import pytest
import torch
from safetensors import safe_open

from sglang.test.comparison_test.common import *
from sglang.test.comparison_test.tensor_tracer import tracing_enabled
from sglang.test.comparison_test.tf.load_weights import (
    load_random_weights,
    load_weight_from_hf_ckp,
    save_model_weights,
)
from sglang.test.comparison_test.tf.test_moe_gate import MoEGate

# todo
MoEGate_configs = [
    {
        "num_experts_per_tok": 8,
        "n_routed_experts": 96,
        "routed_scaling_factor": 1.0,
        "scoring_func": "sigmoid",
        "seq_aux": True,
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "hidden_size": 5120,
    }
]
# Define the real weight prefixes in the model checkpoint for the MoEGate tests
real_weight_prefixs = ["model.layers.0.mlp.gate", "model.layers.62.mlp.gate"]
# Define the random weight test count as the same as the real weight test count
random_weights = [0] * len(real_weight_prefixs)


@torch.inference_mode()
def _run_moe_gate_random_input(moe_gate, dtype, log_dir):
    """Run the MoEGate with random input and trace tensors."""
    moe_gate = moe_gate.to(dtype=dtype).cuda()
    weight_file = os.path.join(log_dir, WEIGHTS_FILE)
    # save the weights to a file
    save_model_weights(moe_gate, weight_file)

    with tracing_enabled(verbose=False) as tracer:
        for bs in BATCH_SIZES:
            for sl in SEQ_LENS:
                print(f"Testing MoEGate with real weights: {bs=} {sl=}")
                # Create a random input tensor
                input_tensor = torch.randn(bs, sl, moe_gate.gating_dim, dtype=dtype)
                for _ in range(REPEAT_COUNT):
                    print(f"    Repeat {_+1}/{REPEAT_COUNT}")
                    moe_gate(input_tensor)
                # Save the traced tensors
                saved_path = os.path.join(log_dir, f"traced_tensor_{bs=}_{sl=}")
                tracer.save_traced_tensors(saved_path)
                print(f"Traced tensors saved to {saved_path}")


@pytest.mark.parametrize("config", MoEGate_configs)
@pytest.mark.parametrize("real_weight_prefix", real_weight_prefixs)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_moe_gate_load_weights(config, real_weight_prefix, dtype):
    """Test MoEGate with real weights loaded from a checkpoint."""
    # Prepare the log directory to save the weight and traced tensors as benchmark
    log_dir = os.path.join(
        LOG_DIR,
        "tf",
        "moe_gate",
        f"{config['hidden_size']}",
        f"real_weights_{real_weight_prefix}",
    )
    os.makedirs(log_dir, exist_ok=True)
    test_config = ComparisonTestConfig(
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

    # Initialize the MoEGate with the given configuration
    moe_gate = MoEGate(config)
    load_weight_from_hf_ckp(moe_gate, test_config, real_weight_prefix, dtype=dtype)
    # Run the MoEGate with real weights and trace tensors
    print(
        f"Testing MoEGate with real weights: {config=} "
        f"{real_weight_prefix=} {dtype=}"
    )
    _run_moe_gate_random_input(moe_gate, dtype, log_dir)


@pytest.mark.parametrize("config", MoEGate_configs)
@pytest.mark.parametrize("random_weight", random_weights)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_moe_gate_random_weights(config, random_weight, dtype):
    """Test MoEGate with random weights."""
    # Prepare the log directory to save the weight and traced tensors as benchmark
    log_dir = os.path.join(
        LOG_DIR,
        "tf",
        "moe_gate",
        f"{config['hidden_size']}_{config['num_experts_per_tok']}_{config['n_routed_experts']}",
        f"random_weights_{uuid.uuid4().hex[:8]}",
    )
    os.makedirs(log_dir, exist_ok=True)
    test_config = ComparisonTestConfig(
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

    # Initialize the MoEGate with the given configuration
    moe_gate = MoEGate(config)
    load_random_weights(moe_gate, dtype=dtype)

    # Run the MoEGate with random weights and trace tensors
    print(
        f"Testing MoEGate with random weights: {config=} " f"{random_weight=} {dtype=}"
    )
    _run_moe_gate_random_input(MoEGate, dtype, log_dir)
