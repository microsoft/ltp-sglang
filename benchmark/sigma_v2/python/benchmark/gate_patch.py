import sglang.srt.layers.moe.topk as moe_topk
import torch
from typing import Optional
from sglang.srt.distributed import get_tensor_model_parallel_world_size

def biased_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    compiled: bool = True,
    n_share_experts_fusion: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[object] = None,
):
    G = get_tensor_model_parallel_world_size()
    K = hidden_states.shape[0] * G
    E = gating_output.shape[1]
    EG = num_expert_group
    EXPERTS_PER_GROUP = E // EG
    EXPERTS_PER_GPU = E // G
    TOKENS_PER_GPU = K // G
    assert K % G == 0
    assert E % G == 0

    device = hidden_states.device
    expert_to_gpu = torch.div(torch.arange(E, device=device), EXPERTS_PER_GPU, rounding_mode='floor')

    # Precompute expert groups
    expert_ids = torch.arange(E, device=device)
    group_ids = expert_ids // EXPERTS_PER_GROUP
    group_to_experts = [expert_ids[group_ids == g] for g in range(EG)]

    topk_idx = torch.empty((TOKENS_PER_GPU, topk), dtype=torch.long, device=device)
    topk_weight = torch.full((TOKENS_PER_GPU, topk), 1.0 / topk, dtype=torch.float32, device=device)

    # GPU load map (per expert GPU)
    gpu_load = torch.zeros(G, device=device, dtype=torch.int32)

    for token_idx in range(TOKENS_PER_GPU):
        base_group = token_idx % EG
        selected_experts = []

        for offset in range(4):
            group_id = (base_group + offset) % EG
            candidates = group_to_experts[group_id]

            # Get GPU loads for each expert in group
            expert_gpus = expert_to_gpu[candidates]
            loads = gpu_load[expert_gpus]

            # Get top 2 experts with lowest load
            _, indices = torch.topk(-loads, 2)  # negative for lowest
            chosen = candidates[indices]

            selected_experts.append(chosen)
            gpu_load.index_add_(0, expert_to_gpu[chosen], torch.ones(2, dtype=torch.int32, device=device))

        # Flatten the 4x2 list into 1x8
        topk_idx[token_idx] = torch.cat(selected_experts)

    return topk_weight, topk_idx

original_biased_grouped_topk = moe_topk.biased_grouped_topk
moe_topk.biased_grouped_topk = biased_grouped_topk