import torch


def compare_res(res1, res2, tolerance=1e-5):
    """Compare two res and return the maximum absolute difference."""
    if type(res1) != type(res2):
        raise ValueError(f"Results have different types: {type(res1)} vs {type(res2)}")
    if isinstance(res1, torch.Tensor):
        return compare_tensor_pair(res1, res2, tolerance=tolerance)
    elif isinstance(res1, list):
        if len(res1) != len(res2):
            raise ValueError(
                f"Results have different lengths: {len(res1)} vs {len(res2)}"
            )
        return [compare_res(r1, r2, tolerance=tolerance) for r1, r2 in zip(res1, res2)]
    elif isinstance(res1, dict):
        if set(res1.keys()) != set(res2.keys()):
            raise ValueError(
                f"Results have different keys: {set(res1.keys())} vs {set(res2.keys())}"
            )
        return {
            k: compare_res(res1[k], res2[k], tolerance=tolerance) for k in res1.keys()
        }
    elif isinstance(res1, tuple):
        if len(res1) != len(res2):
            raise ValueError(
                f"Results have different lengths: {len(res1)} vs {len(res2)}"
            )
        return [compare_res(r1, r2, tolerance=tolerance) for r1, r2 in zip(res1, res2)]
    else:
        print(f"Comparing non-tensor results: {type(res1)}")
        if res1 != res2:
            raise ValueError(f"Results differ: {res1} vs {res2}")
        print(f"Results are the same: {type(res1)}: {res1}")
        return None, None  # No difference found


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


def calculate_mean_std(tensors: list[torch.Tensor]) -> tuple:
    """Calculate the mean and standard deviation of each dimension in a tensor list."""
    if not tensors:
        raise ValueError("The list of tensors is empty.")

    # Stack tensors to create a 3D tensor (num_tensors, num_elements, num_features)
    stacked_tensors = torch.stack(tensors)

    # Calculate mean and std across the first dimension (num_tensors)
    mean = stacked_tensors.mean(dim=0)
    std = stacked_tensors.std(dim=0)

    return mean, std


def compare_two_distributions(tensor_dis1, tensor_dis2, tolerance=1e-5):
    """Compare two distributions of tensors/results, calculate their difference of mean and std."""
    mean1, std1 = calculate_mean_std(tensor_dis1)
    mean2, std2 = calculate_mean_std(tensor_dis2)

    mean_diff = torch.abs(mean1 - mean2)
    std_diff = torch.abs(std1 - std2)

    if mean_diff.max().item() > tolerance:
        raise ValueError(f"Mean difference exceeds tolerance: {mean_diff}")
    if std_diff.max().item() > tolerance:
        raise ValueError(f"Standard deviation difference exceeds tolerance: {std_diff}")

    # TODO: get the distribution of differences
    return mean_diff, std_diff
