import json
import os
import uuid

import pytest
import torch
from safetensors.torch import safe_open, save_file

from sglang.test.comparison_test.common import *
from sglang.test.comparison_test.tensor_tracer import tracing_enabled
from sglang.test.comparison_test.tf.load_weights import (
    load_random_weights,
    load_weight_from_hf_ckp,
    save_model_weights,
)
from sglang.test.comparison_test.tf.test_mlp import MLP

# Define the configurations for the MLP tests
MLP_configs = [{"hidden_size": 5120, "intermediate_size": 2160, "hidden_act": "silu"}]
# Define the real weight prefixes in the model checkpoint for the MLP tests
real_weight_prefixs = [
    "model.layers.0.mlp.experts.0",
    "model.layers.62.mlp.experts.95",
]
# Define the random weight test count as the same as the real weight test count
random_weights = [0] * len(real_weight_prefixs)


@torch.inference_mode()
def _run_mlp_random_input(mlp, dtype, log_dir):
    """Run the MLP with random input and trace tensors."""
    mlp = mlp.to(dtype=dtype).cuda()
    weight_file = os.path.join(log_dir, WEIGHTS_FILE)
    # save the weights to a file
    save_model_weights(mlp, weight_file)

    with tracing_enabled(verbose=False) as tracer:
        for bs in BATCH_SIZES:
            for sl in SEQ_LENS:
                print(f"Testing MLP with real weights: {bs=} {sl=}")
                # Create a random input tensor
                input_tensor = torch.randn(bs, sl, mlp.hidden_size, dtype=dtype).cuda()
                for _ in range(REPEAT_COUNT):
                    print(f"    Repeat {_+1}/{REPEAT_COUNT}")
                    mlp(input_tensor)
                # Save the traced tensors
                saved_path = os.path.join(log_dir, f"traced_tensor_{bs=}_{sl=}")
                tracer.save_traced_tensors(saved_path)
                print(f"Traced tensors saved to {saved_path}")


@pytest.mark.parametrize("config", MLP_configs)
@pytest.mark.parametrize("real_weight_prefix", real_weight_prefixs)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_mlp_load_weights(config, real_weight_prefix, dtype):
    """Test MLP with real weights loaded from a checkpoint."""

    # Prepare the log directory to save the weight and traced tensors as benchmark
    log_dir = os.path.join(
        LOG_DIR,
        "tf",
        "mlp",
        f"{config['hidden_size']}_{config['intermediate_size']}_{config['hidden_act']}",
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

    # Initialize the MLP with the given configuration
    mlp = MLP(config)
    load_weight_from_hf_ckp(mlp, real_weight_prefix)

    # Run the MLP with real weights and trace tensors
    print(
        f"Testing MLP with real weights: {config=} " f"{real_weight_prefix=} {dtype=}"
    )
    _run_mlp_random_input(mlp, dtype, log_dir)


@pytest.mark.parametrize("config", MLP_configs)
@pytest.mark.parametrize("random_weight", random_weights)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_mlp_random_weights(config, random_weight, dtype):
    """Test MLP with random weights."""
    # Prepare the log directory to save the weight and traced tensors as benchmark
    log_dir = os.path.join(
        LOG_DIR,
        "tf",
        "mlp",
        f"{config['hidden_size']}_{config['intermediate_size']}_{config['hidden_act']}",
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

    # Initialize the MLP with the given configuration
    mlp = MLP(config)
    load_random_weights(mlp)

    # Run the MLP with random weights and trace tensors
    print(f"Testing MLP with random weights: {config} {dtype}")
    _run_mlp_random_input(mlp, dtype, log_dir)
