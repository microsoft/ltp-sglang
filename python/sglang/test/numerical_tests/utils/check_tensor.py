# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


def compare_tensor_pair(tensor1: torch.Tensor, tensor2: torch.Tensor, tolerance=1e-5):
    """Compare two tensors and return the maximum absolute difference."""
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            f"Tensors have different shapes: {tensor1.shape} vs {tensor2.shape}"
        )
    return (
        torch.max(torch.abs(tensor1 - tensor2)).item(),
        torch.allclose(tensor1, tensor2, atol=tolerance),
        torch.nn.functional.cosine_similarity(
            tensor1.view(-1).double(), tensor2.view(-1).double(), dim=0
        ).item(),
    )


def compare_tensors(tensor_list: list[torch.Tensor], tolerance=1e-5):
    """Compare a list of tensors and return the maximum absolute difference."""
    if not tensor_list:
        raise ValueError("The list of tensors is empty.")

    max_diffs = []
    is_closes = []
    cos_similarities = []
    # Compare each pair of tensors in the list
    for i in range(len(tensor_list) - 1):
        for j in range(i + 1, len(tensor_list)):
            diff, is_close, cos_similarity = compare_tensor_pair(
                tensor_list[i], tensor_list[j], tolerance=tolerance
            )
            max_diffs.append(diff)
            is_closes.append(is_close)
            cos_similarities.append(cos_similarity)

    return max_diffs, is_closes, cos_similarities


def get_mean_std(tensor_list: list[torch.Tensor]):
    """Calculate the mean and standard deviation of a list of tensors."""
    if not tensor_list:
        raise ValueError("The list of tensors is empty.")

    # Ensure all tensors are on the same device and have the same shape
    first_tensor = tensor_list[0]
    for tensor in tensor_list[1:]:
        if tensor.shape != first_tensor.shape or tensor.device != first_tensor.device:
            raise ValueError(
                "All tensors must have the same shape and be on the same device."
            )

    # Stack tensors and calculate mean and std
    stacked_tensors = torch.stack(tensor_list)
    mean = torch.mean(stacked_tensors, dim=0)
    std = torch.std(stacked_tensors, dim=0)
    return mean, std


def compare_output_lists(bench_output, sglang_output):
    """Compare two lists of tensors and return the maximum absolute difference."""
    if len(bench_output) != len(sglang_output):
        raise ValueError("Output lists have different lengths.")

    bench_mean, bench_std = get_mean_std(bench_output)
    sglang_mean, sglang_std = get_mean_std(sglang_output)

    max_mean_diff, _, similarity = compare_tensor_pair(bench_mean, sglang_mean)
    max_std_diff = torch.max(torch.abs(bench_std - sglang_std)).item()

    return similarity, max_mean_diff, max_std_diff
