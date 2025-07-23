import json
import os

import torch
from safetensors import safe_open
from torch import nn

from sglang.test.comparison_test.common import CHECKPOINT_PATH


def load_random_weights(module: nn.Module, dtype=torch.bfloat16):
    """Initialize weights for a given layer."""
    for name, param in module.named_parameters():
        if "weight" in name:
            param.data = torch.randn_like(param.data, dtype=dtype)
        elif "bias" in name and param is not None:
            param.data = torch.randn_like(param.data, dtype=dtype)


def load_weight_from_hf_ckp(
    module: nn.Module, layer_prefix: str, ckp_path: str = CHECKPOINT_PATH
):
    """Initialize weights from a huggingface checkpoint folder according to the fields in the given nn.Module."""
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
    # the the fields of the module
    fields = [name for name, _ in module.named_parameters()]
    file_to_fields = {}

    for field in fields:
        layer_key = f"{layer_prefix}.{field}"
        if layer_key not in weight_map:
            raise KeyError(f"Weight key {layer_key} not found in index file.")
        # Get the files containing our weights
        tensor_file = weight_map[layer_key]
        file_to_fields.setdefault(tensor_file, []).append(layer_key)

    # Load the weights from the safetensors files
    for file_name, layer_keys in file_to_fields.items():
        with safe_open(file_name, framework="pt", device="cpu") as f:
            for key in layer_keys:
                if key in f.keys():
                    # Remove the layer prefix to get the local field name
                    field_name = key.replace(f"{layer_prefix}.", "")
                    param = dict(module.named_parameters())[field_name]
                    param.data = f.get_tensor(key)
                    print(f"Loaded {key} from {file_name}")
                else:
                    raise ValueError(f"Weight {key} not found in {file_name}")

    return module
