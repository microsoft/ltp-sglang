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
