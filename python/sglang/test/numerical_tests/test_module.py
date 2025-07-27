import json
import os

import torch
from torch import nn

from sglang.srt.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.utils import set_random_seed
from sglang.test.numerical_tests.common import *
from sglang.test.numerical_tests.utils.tensor_checker import compare_tensors


class TestModule:
    """
    Test general modules
    """

    def setup_method(self, method):
        self.setup_distributed()
        self.setup_parallelism()
        self.setup_random_seed()

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
            pipeline_model_parallel_size=1,
        )

    def setup_random_seed(self):
        random_seed = 42
        set_random_seed(random_seed)

    @torch.inference_mode()
    def _run_module_random_input(
        self,
        forward_func: callable,
        random_input_func: callable,
        dtype: torch.dtype,
        log_dir: str,
    ):
        """Run the MoE with random input and trace tensors."""

        results = {}
        for bs in BATCH_SIZES:
            for sl in SEQ_LENS:
                print(f"    Testing MoE with random input: {bs=} {sl=}")
                all_max_diffs = []
                all_is_closes = []
                all_cos_similarities = []
                for i in range(RANDOM_INPUT_COUNT):
                    # Create a random input tensor
                    # XXX: ignore batch size for now
                    inputs = random_input_func(bs, sl, dtype=dtype)
                    out_tensors = []
                    assert (
                        REPEAT_COUNT > 1
                    ), "REPEAT_COUNT should be greater than 1 to compare outputs"
                    for j in range(REPEAT_COUNT):
                        # Call the forward function
                        out_tensor = forward_func(inputs)
                        out_tensors.append(out_tensor.detach().cpu())
                    # Compare the outputs
                    max_diffs, is_closes, cos_similarities = compare_tensors(
                        out_tensors
                    )
                    all_max_diffs.extend(max_diffs)
                    all_is_closes.extend(is_closes)
                    all_cos_similarities.extend(cos_similarities)
                # Save the results
                results[f"{bs=}-{sl=}"] = {
                    "max_diffs": all_max_diffs,
                    "is_closes": all_is_closes,
                    "cos_similarities": all_cos_similarities,
                    "cosine_similarity_range": (
                        min(all_cos_similarities),
                        max(all_cos_similarities),
                    ),
                    "max_diff_range": (min(all_max_diffs), max(all_max_diffs)),
                }
                print(f"    Results for {bs=}-{sl=}:")
                print(
                    f"        Cosine Similarity Range: "
                    f"{results[f'{bs=}-{sl=}']['cosine_similarity_range']}"
                )
                print(
                    f"        Max Diff Range: {results[f'{bs=}-{sl=}']['max_diff_range']}"
                )
        # Save the results to a file
        with open(os.path.join(log_dir, RESULTS_FILE), "w") as f:
            json.dump(results, f, indent=4)

    def teardown_method(self, method):
        destroy_model_parallel()
        destroy_distributed_environment()
