import json
import os

import torch

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
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

    def setup_method(self, method):
        self.setup_distributed()
        self.setup_parallelism()

    def setup_distributed(self):
        init_distributed_environment(
            backend="nccl",
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="env://127.0.0.1:29500",
        )

    def setup_parallelism(self):
        initialize_model_parallel(
            tensor_model_parallel_size=1,
        )

    def teardown_method(self, method):
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
                f"Testing {sgl_module.__class__.__name__} in {sub_folder} with config: {bench_config.module_config}"
            )
            compare_results = {}
            for tensors_info_dict, tensor_folder in load_all_input_output(sub_folder):
                # In each tensor folder, the input tensors have the same shape and type
                print(f"Loaded tensors from {tensor_folder}")
                trace_metadata = TraceMetadata.from_dict(tensors_info_dict)
                tensor_folder_name = os.path.basename(tensor_folder)

                all_similiarities = []
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

                    all_similiarities.append(simi)
                    all_max_mean_diffs.append(max_mean_diff)
                    all_max_std_diffs.append(max_std_diff)

                # Store the comparison results for this tensor folder
                # which contains multiple trace groups with the same shape
                compare_results[tensor_folder_name] = {
                    "similarity": all_similiarities,
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
