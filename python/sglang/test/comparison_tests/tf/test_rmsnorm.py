import json
import os
import uuid

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from sglang.test.comparison_tests.common import *
from sglang.test.comparison_tests.tensor_tracer import trace_tensors, tracing_enabled
from sglang.test.comparison_tests.tf.load_weights import (
    load_random_weights,
    load_weight_from_hf_ckp,
)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @trace_tensors("RMSNorm")
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


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
