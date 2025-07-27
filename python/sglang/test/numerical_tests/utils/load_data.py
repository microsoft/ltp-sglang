import json
import os

import torch
from safetensors import safe_open
from torch import nn

from sglang.test.numerical_tests.common import CHECKPOINT_PATH


def load_random_weights(sgl_module: nn.Module, dtype=torch.bfloat16):
    """Load random weights into a sglang module."""
    for name, param in sgl_module.named_parameters():
        if "weight" in name:
            param.data = torch.randn_like(param.data, dtype=dtype)
        elif "bias" in name and param is not None:
            param.data = torch.randn_like(param.data, dtype=dtype)
    # Set the module to evaluation mode
    sgl_module = sgl_module.to(dtype=dtype).cuda()
    sgl_module.eval()
    return sgl_module


def load_weight_from_hf_ckp(
    sgl_module: nn.Module,
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
    if hasattr(sgl_module, "load_weights"):
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
        sgl_module.load_weights(weights)
        # Move the module to the specified dtype and GPU
        sgl_module = sgl_module.to(dtype=dtype).cuda()
        sgl_module.eval()
        return sgl_module

    # the the fields of the module
    fields = [name for name, _ in sgl_module.named_parameters()]

    for field in fields:
        layer_key = f"{layer_prefix}.{field}"
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
                    field_name = key.replace(f"{layer_prefix}.", "")
                    param = dict(sgl_module.named_parameters())[field_name]
                    param.data = f.get_tensor(key)
                else:
                    raise ValueError(f"Weight {key} not found in {file_name}")
    # Move the module to the specified dtype and GPU
    sgl_module = sgl_module.to(dtype=dtype).cuda()
    sgl_module.eval()

    return sgl_module
