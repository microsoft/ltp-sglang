# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os

import torch

from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.utils.common import get_device

from sglang.test.numerical_tests.bench_module import BenchConfig, TraceMetadata
from sglang.test.numerical_tests.utils.check_tensor import compare_output_lists
from sglang.test.numerical_tests.utils.common import (
    BENCHMARK_FOLDER,
    COMPARE_RESULTS_FILE,
    LOG_DIR,
    REPEAT_COUNT,
)
from sglang.test.numerical_tests.utils.load_data import (
    find_all_benchmark_folders,
    load_all_input_output,
    load_module,
    load_test_config,
)


class CompareModule:
    
    @classmethod
    def setup_class(cls):
        """Setup once for the entire test class."""
        cls.setup_distributed()
        cls.setup_parallelism()
        device = get_device()
        server_args = ServerArgs(model_path="dummy", device=device)
        set_global_server_args_for_scheduler(server_args)

    @classmethod
    def setup_distributed(cls):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method="tcp://127.0.0.1:23456",
                world_size=1,
                rank=0,
            )
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:23456",
            local_rank=0,
            backend="nccl" if torch.cuda.is_available() else "gloo",
        )

    @classmethod
    def setup_parallelism(cls):
        initialize_model_parallel(
            tensor_model_parallel_size=1,
        )

    @classmethod
    def teardown_class(cls):
        """Cleanup once after all tests in the class complete."""
        destroy_model_parallel()
        destroy_distributed_environment()


    def _test_module_comparison(
        self,
        module_init_func,
        module_forward_func,
        dtype=torch.bfloat16,
        bench_dir=BENCHMARK_FOLDER,
        log_dir=LOG_DIR,
        module_prefix="",
    ):
        bench_folders = find_all_benchmark_folders(bench_dir)
        all_results = {}
        for sub_folder in bench_folders:
            bench_config = load_test_config(sub_folder)
            bench_config = BenchConfig(**bench_config)
            sgl_module = module_init_func(bench_config.module_config)

            sgl_module = load_module(
                sgl_module, sub_folder, dtype=dtype, module_prefix=module_prefix
            )
            print(
                f"Testing {sgl_module.__class__.__name__} in {sub_folder} "
                f"with config: {bench_config.module_config}"
            )
            compare_results = {}
            for tensors_info_dict, tensor_folder in load_all_input_output(sub_folder):
                # In each tensor folder, the input tensors have the same shape and type
                print(f"Loaded tensors from {tensor_folder}")
                trace_metadata = TraceMetadata.from_dict(tensors_info_dict)
                tensor_folder_name = os.path.basename(tensor_folder)

                all_similarities = []
                all_max_mean_diffs = []
                all_max_std_diffs = []
                for trace_group in trace_metadata.groups:
                    input_tensor_names = trace_group.input_tensors
                    bench_output_tensor_names = trace_group.output_tensors
                    # Record the output tensors from the sglang module
                    sglang_output_tensors = []
                    for _ in range(REPEAT_COUNT):
                        # Load the input tensors for the current trace group
                        inputs = {
                            name: torch.load(os.path.join(tensor_folder, file)).cuda()
                            for name, file in input_tensor_names.items()
                        }
                        # Forward pass with the input tensors
                        output = module_forward_func(sgl_module, inputs, trace_metadata)
                        sglang_output_tensors.append(output)
                    # Load the benchmark output tensors
                    bench_output_tensors = [
                        torch.load(os.path.join(tensor_folder, file)).to(dtype).cuda()
                        for file in bench_output_tensor_names
                    ]
                    # Compare the output tensors between the benchmark and sglang module
                    simi, max_mean_diff, max_std_diff = compare_output_lists(
                        bench_output_tensors, sglang_output_tensors
                    )

                    all_similarities.append(round(simi, 4))
                    all_max_mean_diffs.append(round(max_mean_diff, 4))
                    all_max_std_diffs.append(round(max_std_diff, 4))

                # Store the comparison results for this tensor folder
                # which contains multiple trace groups with the same shape
                compare_results[tensor_folder_name] = {
                    "similarity": all_similarities,
                    "max_mean_diff": all_max_mean_diffs,
                    "max_std_diff": all_max_std_diffs,
                }

            # Store the comparison results for this benchmark folder for one specific module
            all_results[sub_folder] = compare_results

        # Save the results to a file
        os.makedirs(log_dir, exist_ok=True)
        results_file = os.path.join(log_dir, COMPARE_RESULTS_FILE)
        with open(results_file, "w") as f:
            json.dump(all_results, f)
            print(f"Comparison results saved to {results_file}")
