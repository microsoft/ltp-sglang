import json
import os

import torch
from torch import nn

from sglang.test.comparison_test.common import TEST_CONFIG_FILE, WEIGHTS_FILE
from sglang.test.comparison_test.tensor_tracer import TensorGroup


def load_weights(sgl_module: nn.Module, weight_path: str, dtype=torch.bfloat16):
    """Load weights from a safetensors file into a sglang module."""
    from safetensors import safe_open

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight path {weight_path} does not exist.")

    with safe_open(weight_path, framework="pt", device="cpu") as f:
        for name, param in sgl_module.named_parameters():
            if name in f.keys():
                param.data = f.get_tensor(name)
                print(f"Loaded {name} from {weight_path}")
            else:
                raise KeyError(f"Weight {name} not found in {weight_path}")

    sgl_module = sgl_module.to(dtype=dtype).cuda()
    return sgl_module


def load_tf_data(sgl_module: nn.Module, data_folder, dtype=torch.bfloat16):
    """Load weights and other data from a TensorFlow checkpoint folder."""
    import json

    weight_file = os.path.join(data_folder, WEIGHTS_FILE)
    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"Weight file {weight_file} does not exist.")

    sgl_module = load_weights(sgl_module, weight_file, dtype=dtype)

    test_config = {}
    test_config_file = os.path.join(data_folder, TEST_CONFIG_FILE)
    if os.path.exists(test_config_file):
        with open(test_config_file, "r") as f:
            test_config = json.load(f)
        print(f"Loaded configuration: {test_config}")

    return sgl_module, test_config


def load_input_output(traced_tensor_folder: str):
    """Load input and output tensors from the traced tensor folder."""
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
