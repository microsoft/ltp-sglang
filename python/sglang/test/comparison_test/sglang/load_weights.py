import os

import torch
from torch import nn

from sglang.test.comparison_test.common import TEST_CONFIG_FILE, WEIGHTS_FILE


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


def load_input_output(pt_file: str):
    """Load input and output tensors from a PyTorch file."""
    import torch

    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"PyTorch file {pt_file} does not exist.")

    with torch.no_grad():
        data = torch.load(pt_file, map_location="cpu")
        input_tensor = data.get("input_tensor")
        output_tensor = data.get("output_tensor")

    if input_tensor is None or output_tensor is None:
        raise ValueError("Input or output tensor not found in the file.")

    return input_tensor, output_tensor
