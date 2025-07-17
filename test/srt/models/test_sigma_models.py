# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Usage:

To test a specific model locally:
1. Add it to ALL_MODELS, for example, `ModelCase("/path/to/model", tp_size=1)`.
2. Run `ONLY_RUN=/path/to/model python3 -m unittest test_sigma_models.TestGenerationModels`
"""

import argparse
import dataclasses
import multiprocessing as mp
import os
import random
import sys
import unittest
from typing import List

import torch

from sglang.test.runners import (
    DEFAULT_PROMPTS,
    HFRunner,
    SRTRunner,
    check_close_model_outputs,
)
from sglang.test.test_utils import CustomTestCase, is_in_ci


@dataclasses.dataclass
class ModelCase:
    model_path: str
    tp_size: int = 1
    attention_backend: str = "triton"
    repeat: int = 10
    prefill_tolerance: float = 5e-2
    decode_tolerance: float = 5e-2
    rouge_l_tolerance: float = 1
    skip_long_prompt: bool = False
    trust_remote_code: bool = False


# the complete set of models to test sglang's generation model
def get_all_models_from_args():
    # Example usage: python3 -m unittest test_sigma_models.TestGenerationModels --model_path=/path/to/model --tp_size=8
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="/paidata/inference-ckp/sigma-200b"
    )
    parser.add_argument("--tp_size", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--attention_backend", type=str, default="triton")
    parser.add_argument("--prefill_tolerance", type=float, default=5e-2)
    parser.add_argument("--decode_tolerance", type=float, default=5e-2)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_long_prompt", action="store_true")
    args, _ = parser.parse_known_args()

    return [
        ModelCase(
            args.model_path,
            tp_size=args.tp_size,
            repeat=args.repeat,
            attention_backend=args.attention_backend,
            prefill_tolerance=args.prefill_tolerance,
            decode_tolerance=args.decode_tolerance,
            trust_remote_code=args.trust_remote_code,
            skip_long_prompt=args.skip_long_prompt,
        )
    ]


ALL_MODELS = get_all_models_from_args()

TORCH_DTYPES = [torch.bfloat16]


class TestGenerationModels(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def assert_close_logits_and_output_strs(
        self,
        prompts: List[str],
        model_case: ModelCase,
        torch_dtype: torch.dtype,
    ) -> None:
        model_path = model_case.model_path
        prefill_tolerance, decode_tolerance, rouge_l_tolerance = (
            model_case.prefill_tolerance,
            model_case.decode_tolerance,
            model_case.rouge_l_tolerance,
        )
        max_new_tokens = 32

        # repeated_srt_outputs = []
        first_srt_outputs = None
        with SRTRunner(
            model_path,
            tp_size=model_case.tp_size,
            torch_dtype=torch_dtype,
            model_type="generation",
            trust_remote_code=model_case.trust_remote_code,
            attention_backend=model_case.attention_backend,
        ) as srt_runner:
            for i in range(model_case.repeat):  # Repeat to get stable outputs
                srt_outputs = srt_runner.forward(prompts, max_new_tokens=max_new_tokens)
                # repeated_srt_outputs.append(srt_outputs)
                if first_srt_outputs is None:
                    first_srt_outputs = srt_outputs
                else:
                    print(f"Comparing SRT outputs for repeat {i+1}/{model_case.repeat}")
                    check_close_model_outputs(
                        hf_outputs=first_srt_outputs,
                        srt_outputs=srt_outputs,
                        prefill_tolerance=model_case.prefill_tolerance,
                        decode_tolerance=model_case.decode_tolerance,
                        rouge_l_tolerance=model_case.rouge_l_tolerance,
                        debug_text=f"model_path={model_path} prompts={prompts}",
                    )

    @unittest.skipIf(is_in_ci(), "CI only runs selected models for simplicity")
    def test_all_models(self):
        for model_case in ALL_MODELS:
            for torch_dtype in TORCH_DTYPES:
                if (
                    "ONLY_RUN" in os.environ
                    and os.environ["ONLY_RUN"] != model_case.model_path
                ):
                    continue

                # Skip long prompts for models that do not have a long context
                prompts = DEFAULT_PROMPTS
                if model_case.skip_long_prompt:
                    prompts = [p for p in DEFAULT_PROMPTS if len(p) < 1000]

                # Assert the logits and output strs are close
                self.assert_close_logits_and_output_strs(
                    prompts, model_case, torch_dtype
                )


if __name__ == "__main__":
    unittest.main()
