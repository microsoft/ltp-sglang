import json
import os

import torch
from safetensors import safe_open
from torch import nn

from sglang.test.numerical_tests.utils.common import (
    BENCH_CONFIG_FILE,
    CHECKPOINT_PATH,
    TRACED_TENSORS_FILE,
    WEIGHTS_FILE,
)


def load_random_weights(module: nn.Module, dtype=torch.bfloat16):
    """Load random weights into a sglang module."""
    for name, param in module.named_parameters():
        if "weight" in name:
            param.data = torch.randn_like(param.data, dtype=dtype)
        elif "bias" in name and param is not None:
            param.data = torch.randn_like(param.data, dtype=dtype)
    # Set the module to evaluation mode
    module = module.to(dtype=dtype).cuda()
    module.eval()
    return module


def load_weight_from_hf_ckp(
    module: nn.Module,
    layer_prefix: str,
    dtype=torch.bfloat16,
    ckp_path: str = CHECKPOINT_PATH,
):
    """Load weights from a Hugging Face checkpoint into a sglang module."""
    if not os.path.exists(ckp_path):
        raise FileNotFoundError(f"Checkpoint path {ckp_path} does not exist.")

    # Load the index file to find which safetensor file contains our layer
    index_path = os.path.join(ckp_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file {index_path} does not exist.")

    with open(index_path, "r") as f:
        index = json.load(f)

    # Find the relevant weight keys for this layer
    weight_map = index["weight_map"]
    file_to_fields = {}

    # Check whether the module has the 'load_weights' method
    # If it has, use it to load the weights
    if hasattr(module, "load_weights"):
        for weight_name in weight_map:
            if weight_name.startswith(layer_prefix):
                # Remove the layer prefix to get the local field name
                field_name = weight_name.replace(f"{layer_prefix}.", "")
                file_to_fields.setdefault(weight_map[weight_name], []).append(
                    weight_name
                )
        # Load the weights from the safetensors files
        weights = []
        for file_name, fields in file_to_fields.items():
            with safe_open(
                os.path.join(ckp_path, file_name), framework="pt", device="cpu"
            ) as f:
                for field in fields:
                    if field in f.keys():
                        weights.append(
                            (field.replace(f"{layer_prefix}.", ""), f.get_tensor(field))
                        )
                    else:
                        raise ValueError(f"Weight {field} not found in {file_name}")
        module.load_weights(weights)
        # Move the module to the specified dtype and GPU
        module = module.to(dtype=dtype).cuda()
        module.eval()
        return module

    # the the fields of the module
    fields = [name for name, _ in module.named_parameters()]

    for field in fields:
        layer_key = f"{layer_prefix}.{field}" if layer_prefix else field
        if layer_key not in weight_map:
            raise KeyError(f"Weight key {layer_key} not found in index file.")
        # Get the files containing our weights
        tensor_file = weight_map[layer_key]
        tensor_file = os.path.join(ckp_path, tensor_file)
        file_to_fields.setdefault(tensor_file, []).append(layer_key)

    # Load the weights from the safetensors files
    for file_name, layer_keys in file_to_fields.items():
        with safe_open(file_name, framework="pt", device="cpu") as f:
            for key in layer_keys:
                if key in f.keys():
                    # Remove the layer prefix to get the local field name
                    field_name = (
                        key.replace(f"{layer_prefix}.", "") if layer_prefix else key
                    )
                    param = dict(module.named_parameters())[field_name]
                    param.data = f.get_tensor(key)
                else:
                    raise ValueError(f"Weight {key} not found in {file_name}")
    # Move the module to the specified dtype and GPU
    module = module.to(dtype=dtype).cuda()
    module.eval()
    print(f"Module {module} loaded with weights from {ckp_path}")
    return module


def load_test_config(data_folder):
    """Load the test configuration from a JSON file."""
    config_file = os.path.join(data_folder, BENCH_CONFIG_FILE)
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Test config file {config_file} does not exist.")

    with open(config_file, "r") as f:
        test_config = json.load(f)

    # Convert dtype string back to torch.dtype
    if "dtype" in test_config:
        test_config["dtype"] = getattr(
            torch, test_config["dtype"].replace("torch.", ""), torch.bfloat16
        )

    return test_config


def find_all_benchmark_folders(base_folder):
    """Find all benchmark folders which contains BENCH_CONFIG_FILE in the base folder."""
    benchmark_folders = []
    for root, _, files in os.walk(base_folder):
        if BENCH_CONFIG_FILE in files:
            benchmark_folders.append(root)
    return benchmark_folders


def load_module(sgl_module: nn.Module, data_folder, dtype=torch.bfloat16):
    """Load weights from a benchmark folder."""

    weight_path = os.path.join(data_folder, WEIGHTS_FILE)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file {weight_path} does not exist.")

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


def load_all_input_output(data_folder):
    """Load all input and output tensors from the benchmark folder."""
    # The traced tensor folders contains TRACED_TENSORS_FILE
    for root, _, files in os.walk(data_folder):
        if TRACED_TENSORS_FILE not in files:
            continue
        with open(os.path.join(root, TRACED_TENSORS_FILE), "r") as f:
            tensors_info = json.load(f)
        # Return the tensors info and the root folder
        # This is a dictionary with id => {input: {name: file}, output: [output_file]}
        # and the root folder is the folder where the traced tensors are stored
        yield tensors_info, root
