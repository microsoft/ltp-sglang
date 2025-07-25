import json
import os

import torch
from torch import nn

from sglang.srt.utils import set_random_seed
from sglang.test.comparison_tests.common import (
    TEST_CONFIG_FILE,
    WEIGHTS_FILE,
    ComparisonTestConfig,
)
from sglang.test.comparison_tests.tensor_tracer import TensorGroup


def load_random_weights(sgl_module: nn.Module, dtype=torch.bfloat16):
    """Load random weights into a sglang module."""
    for name, param in sgl_module.named_parameters():
        if param.requires_grad:
            # Initialize the parameter with random values
            param.data = torch.randn_like(param, dtype=dtype)
            print(f"Initialized {name} with random weights")
    # Set the module to evaluation mode
    sgl_module = sgl_module.to(dtype=dtype).cuda()
    sgl_module.eval()
    return sgl_module


def load_weights(sgl_module: nn.Module, weight_path: str, dtype=torch.bfloat16):
    """Load weights from a safetensors file into a sglang module."""
    from safetensors import safe_open

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight path {weight_path} does not exist.")

    # Check whether the module has the 'load_weights' method
    if hasattr(sgl_module, "load_weights"):
        # If it has, use it to load the weights
        weights = []
        with safe_open(weight_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                weights.append((key, f.get_tensor(key)))
        sgl_module.load_weights(weights)
        # Set the module to evaluation mode
        sgl_module = sgl_module.to(dtype=dtype).cuda()
        return sgl_module

    with safe_open(weight_path, framework="pt", device="cpu") as f:
        for name, param in sgl_module.named_parameters():
            if name in f.keys():
                param.data = f.get_tensor(name)
                print(f"Loaded {name} from {weight_path}")
            else:
                # raise KeyError(f"Weight {name} not found in {weight_path}")
                print(f"Weight {name} not found in {weight_path}")

    sgl_module = sgl_module.to(dtype=dtype).cuda()
    # Set the module to evaluation mode
    sgl_module.eval()
    return sgl_module


def load_input_output(traced_tensor_folder: str):
    """Load input and output tensors from the traced tensor folder.

    Returns:
        tensor_groups (list): List of TensorGroup objects containing input/output tensors.
        tensor_dict (dict): Dictionary mapping tensor names to their data.
    """
    # Load metadata from the traced tensor folder
    metadata_file = os.path.join(traced_tensor_folder, "metadata.json")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file {metadata_file} does not exist.")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    if "groups" not in metadata:
        raise ValueError("Metadata does not contain 'groups' information.")

    # Load tensor groups and their input/output tensors
    tensor_groups = []
    tensor_dict = {}
    for group_name, group_info in metadata["groups"].items():
        input_tensors = group_info.get("input", [])
        output_tensors = group_info.get("output", [])
        tensor_group = TensorGroup(
            name=group_name,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
        )
        tensor_groups.append(tensor_group)
        for tensor_name in input_tensors + output_tensors:
            tensor_file = os.path.join(traced_tensor_folder, f"{tensor_name}.pt")
            if not os.path.exists(tensor_file):
                raise FileNotFoundError(f"Tensor file {tensor_file} does not exist.")
            tensor_data = torch.load(tensor_file)
            tensor_dict[tensor_name] = tensor_data

    return tensor_groups, tensor_dict


def load_module(sgl_module: nn.Module, data_folder, dtype=torch.bfloat16):
    """Load weights from a benchmark folder."""
    import json

    weight_file = os.path.join(data_folder, WEIGHTS_FILE)
    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"Weight file {weight_file} does not exist.")

    sgl_module = load_weights(sgl_module, weight_file, dtype=dtype)

    return sgl_module


def load_test_config(data_folder):
    """Load the test configuration from a JSON file."""
    config_file = os.path.join(data_folder, TEST_CONFIG_FILE)
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Test config file {config_file} does not exist.")

    with open(config_file, "r") as f:
        test_config = json.load(f)

    # Convert dtype string back to torch.dtype
    if "dtype" in test_config:
        test_config["dtype"] = getattr(
            torch, test_config["dtype"].replace("torch.", ""), torch.bfloat16
        )

    return ComparisonTestConfig(**test_config)


def load_all_input_output(data_folder):
    """Load all input and output tensors from the benchmark folder."""
    # The traced tensor folders are named as traced_tensor_bs=1_sl=2
    for root, dirs, _ in os.walk(data_folder):
        for dir_name in dirs:
            if not dir_name.startswith("traced_tensor_"):
                continue
            dir_path = os.path.join(root, dir_name)
            try:
                tensor_groups, tensor_dict = load_input_output(dir_path)
                # yield the tensor groups and their input/output tensors
                yield tensor_groups, tensor_dict, dir_path
            except (FileNotFoundError, ValueError) as e:
                print(f"Error loading tensors from {dir_path}: {e}")
                continue


def find_all_benchmark_folders(base_folder):
    """Find all benchmark folders which contains TEST_CONFIG_FILE in the base folder."""
    benchmark_folders = []
    for root, _, files in os.walk(base_folder):
        if TEST_CONFIG_FILE in files:
            benchmark_folders.append(root)
    return benchmark_folders
