# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import uuid

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import PretrainedConfig

from sglang.test.numerical_tests.utils.common import (
    BENCH_CONFIG_FILE,
    TARGET_DTYPE,
    TRACED_TENSORS_FILE,
    WEIGHTS_FILE,
)


class BenchConfig:
    """Base class for test configurations."""

    module_config: PretrainedConfig = {}
    real_weight_prefix: str = ""
    log_dir: str = ""
    batch_sizes: list[int] = []
    seq_lens: list[int] = []
    dtype: torch.dtype = torch.bfloat16
    input_count: int = 1
    repeat_count: int = 1

    def __init__(
        self,
        module_config,
        real_weight_prefix="",
        log_dir="",
        batch_sizes=[],
        seq_lens=[],
        dtype=torch.bfloat16,
        input_count=1,
        repeat_count=1,
    ):
        """Initialize the test configuration."""
        if isinstance(module_config, dict):
            module_config = PretrainedConfig.from_dict(module_config)
        self.module_config = module_config
        self.real_weight_prefix = real_weight_prefix
        self.log_dir = log_dir
        self.batch_sizes = batch_sizes
        self.seq_lens = seq_lens
        self.dtype = dtype
        self.input_count = input_count
        self.repeat_count = repeat_count

    def save_json(self, path):
        """Save the test configuration to a JSON file."""
        config = {
            "module_config": self.module_config.to_dict(),
            "real_weight_prefix": self.real_weight_prefix,
            "log_dir": self.log_dir,
            "batch_sizes": self.batch_sizes,
            "seq_lens": self.seq_lens,
            "dtype": str(self.dtype),
            "input_count": self.input_count,
            "repeat_count": self.repeat_count,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=4)


class TracedTensorGroup:
    """Class to hold a group of traced tensors."""

    group_id: str
    input_tensors: dict  # {name: file}
    output_tensors: list  # [file1, file2, ...]

    def __init__(self, group_id, input_tensors, output_tensors):
        """Initialize the traced tensor group."""
        self.group_id = group_id
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors


class TraceMetadata:
    """Class to hold metadata for traced tensors."""

    def __init__(self, batch_size, seq_len):
        """Initialize the trace metadata."""
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.groups = list[TracedTensorGroup]()

    def add_group(self, group_id, input_tensors, output_tensors):
        """Add a group of traced tensors."""
        self.groups.append(TracedTensorGroup(group_id, input_tensors, output_tensors))

    def to_dict(self):
        """Convert the trace metadata to a dictionary."""
        return {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "groups": {
                group.group_id: {
                    "input": group.input_tensors,
                    "output": group.output_tensors,
                }
                for group in self.groups
            },
        }

    @classmethod
    def from_dict(cls, data):
        """Create a TraceMetadata instance from a dictionary."""
        metadata = cls(data["batch_size"], data["seq_len"])
        for group_id, tensors in data["groups"].items():
            metadata.add_group(group_id, tensors["input"], tensors["output"])
        return metadata


class BenchModule:
    """
    Base class for testing consistency of SGLang modules
    """

    @torch.inference_mode()
    def _run_module_random_input(
        self,
        module: nn.Module,
        test_config: BenchConfig,
        forward_func: callable,
        random_input_func: callable,
        log_dir: str,
    ):
        """Run the Transformers module with random input and trace tensors."""
        print(f"Running {self.__class__.__name__} with random input to {log_dir}")
        os.makedirs(log_dir, exist_ok=True)

        # Save the weights to a file
        weight_file = os.path.join(log_dir, WEIGHTS_FILE)
        save_model_weights(module, weight_file)

        # Save the test configuration to a JSON file
        config_file = os.path.join(log_dir, BENCH_CONFIG_FILE)
        test_config.save_json(config_file)

        # Set the module to the target dtype and move to GPU
        module = module.to(dtype=TARGET_DTYPE).cuda()

        for bs in test_config.batch_sizes:
            for sl in test_config.seq_lens:
                print(
                    f"Testing {self.__class__.__name__} with random input: {bs=} {sl=}"
                )

                data_folder = os.path.join(log_dir, f"random_input_{bs=}_{sl=}")
                os.makedirs(data_folder, exist_ok=True)

                # Initialize trace metadata
                trace_meta_data = TraceMetadata(batch_size=bs, seq_len=sl)

                for _ in range(test_config.input_count):
                    print(f"    Input {_+1}/{test_config.input_count}")
                    id = uuid.uuid4().hex[:8]
                    # Generate random input tensor
                    inputs = random_input_func(
                        bs, sl, test_config.dtype
                    )  # {name: tensor}
                    traced_inputs = {}
                    for input_tensor_name in inputs:
                        # Clone the input tensor to avoid in-place operations
                        input_tensor_clone = (
                            inputs[input_tensor_name].clone().detach().cpu()
                        )
                        # Save the input tensor
                        input_tensor_file = f"{input_tensor_name}_{id}.pt"
                        traced_inputs[input_tensor_name] = input_tensor_file
                        torch.save(
                            input_tensor_clone,
                            os.path.join(data_folder, input_tensor_file),
                        )

                    repeated_outputs = []
                    # Forward pass
                    for _ in range(test_config.repeat_count):
                        print(f"        Repeat {_+1}/{test_config.repeat_count}")
                        # Forward the module
                        output = forward_func(inputs)
                        # Clone the output tensor to avoid in-place operations
                        output_clone = output.clone().detach().cpu()
                        # Generate a unique name for the output tensor
                        output_tensor_file = f"output_{id}_{_+1}.pt"
                        # Save the output tensor
                        torch.save(
                            output_clone, os.path.join(data_folder, output_tensor_file)
                        )
                        repeated_outputs.append(output_tensor_file)

                    trace_meta_data.add_group(
                        group_id=id,
                        input_tensors=traced_inputs,
                        output_tensors=repeated_outputs,
                    )

                # Save the metadata of traced tensors
                traced_tensors_file = os.path.join(data_folder, TRACED_TENSORS_FILE)
                with open(traced_tensors_file, "w") as f:
                    json.dump(trace_meta_data.to_dict(), f, indent=4)


def save_model_weights(model: torch.nn.Module, output_file: str):
    """
    Save the weights of a PyTorch nn.Module to a .safetensors file.

    Args:
        model (torch.nn.Module): The PyTorch model whose weights you want to save.
        output_file (str): Path to save the weights in `.safetensors` format.

    Example:
        model = MyModel()
        save_model_weights(model, "model_weights.safetensors")
    """
    if not isinstance(model, torch.nn.Module):
        raise ValueError("The model must be an instance of torch.nn.Module.")

    if not output_file.endswith(".safetensors"):
        raise ValueError("The output file must end with '.safetensors'.")

    # Get the state_dict (all weights and buffers from the model)
    state_dict = model.state_dict()

    # Prepare the dictionary for safetensors
    tensors_to_save = {key: tensor for key, tensor in state_dict.items()}

    # Save to a .safetensors file
    save_file(tensors_to_save, output_file)
    print(f"Model weights saved to '{output_file}'")
