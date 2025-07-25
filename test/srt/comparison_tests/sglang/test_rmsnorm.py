import os

import pytest

from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.comparison_tests.common import *
from sglang.test.comparison_tests.sglang.load_data import *
from sglang.test.comparison_tests.tensor_checker import *

RMSNORM_BENCHMARK_FOLDER = os.getenv(
    "RMSNORM_BENCHMARK_FOLDER", "sglang/test/srt/comparison_test/sglang/benchmark"
)


TP_SIZES = [1]


@pytest.mark.parametrize("tp_size", TP_SIZES)
def test_sglang_rmsnorm(tp_size):
    """Run the sglang RMSNorm with random input and trace tensors."""
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

    ben_folders = find_all_benchmark_folders(RMSNORM_BENCHMARK_FOLDER)
    for ben_folder in ben_folders:
        print(f"Testing RMSNorm benchmark folder: {ben_folder}")
        test_config = load_test_config(ben_folder)
        print(f"Loaded test config: {test_config}")

        # Init the module
        sgl_module = (
            RMSNorm(
                test_config.module_config["hidden_size"],
                test_config.module_config["rms_norm_eps"],
            )
            .to(dtype=test_config.dtype)
            .cuda()
        )

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
                max_diff, is_close, cos_similarity = compare_tensors(
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
