import sys

import pytest
import torch
from torch import nn

from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from sglang.srt.layers.moe.ep_moe.layer import EPMoE
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.test.comparison_tests.sglang.load_data import *
from sglang.test.comparison_tests.tensor_checker import *
from sglang.test.numerical_tests.modules.test_moe import MoE

MOE_BENCHMARK_FOLDER = os.getenv(
    "MOE_BENCHMARK_FOLDER", "sglang/test/srt/comparison_test/sglang/benchmark"
)

# MoE Implementation
MoE_modules_impl = [
    FusedMoE,
    # EPMoE,
]
TP_SIZES = [1]


@pytest.mark.parametrize("moe_module_impl", MoE_modules_impl)
@pytest.mark.parametrize("tp_size", TP_SIZES)
def test_sglang_moe(moe_module_impl, tp_size):

    init_distributed_environment(
        backend="nccl",
        world_size=tp_size,
        rank=0,
        local_rank=0,
        distributed_init_method="env://127.0.0.1:29500",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    """Run the sglang MoE with random input and trace tensors."""
    ben_folders = find_all_benchmark_folders(MOE_BENCHMARK_FOLDER)
    for ben_folder in ben_folders:
        print(f"Testing MoE benchmark folder: {ben_folder}")
        test_config = load_test_config(ben_folder)
        print(f"Loaded test config: {test_config}")

        # Init the module
        sgl_module = MoE(test_config.module_config, moe_module_impl)
        sgl_module = load_module(sgl_module, ben_folder, dtype=test_config.dtype)

        # Load the input/output tensors
        for tensor_groups, tensor_dict, traced_tensor_folder in load_all_input_output(
            ben_folder
        ):
            print(f"Loaded tensors from {traced_tensor_folder}")

            output_pair = []  # [(transformer_output, sglang_output), ...]
            # Run the module with the loaded tensors
            for group in tensor_groups:
                input_tensor = tensor_dict[group.input_names[0]]
                output_tensor = tensor_dict[group.output_names[0]]
                # XXX the original input tensor's shape is (bs, sl, hidden_size), bs = 1. I need to reshape it to (sl, hidden_size)
                assert (
                    input_tensor.ndim == 3 and input_tensor.shape[0] == 1
                ), f"Input tensor shape mismatch, expected (1, sl, hidden_size), got {input_tensor.shape}"
                input_tensor = input_tensor.view(-1, input_tensor.shape[-1]).cuda()
                output_tensor = output_tensor.view(-1, output_tensor.shape[-1]).cuda()
                print(
                    f"Running MoE with input shape: {input_tensor.shape} and output shape: {output_tensor.shape}"
                )
                sgl_output = sgl_module(input_tensor)

                # if not torch.allclose(sgl_output, output_tensor, atol=1e-5):
                #     print(
                #         f"Output does not match expected output for group {group.id}:"
                #     )
                output_pair.append((output_tensor, sgl_output))
            # Compare the outputs
            lowest_similarity = 1.0
            max_diff = 0.0
            for expected_output, sgl_output in output_pair:
                max_diff, is_close, cos_similarity = compare_tensor_pair(
                    expected_output, sgl_output
                )
                print(
                    f"Max difference: {max_diff}, Is close: {is_close}, Cosine similarity: {cos_similarity}"
                )
                if cos_similarity < lowest_similarity:
                    lowest_similarity = cos_similarity

            print(f"Lowest cosine similarity: {lowest_similarity}")
            print(f"Max difference: {max_diff}")

    destroy_model_parallel()
    destroy_distributed_environment()
