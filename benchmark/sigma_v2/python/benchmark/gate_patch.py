import torch
from typing import Optional
import sglang.srt.layers.moe.topk as moe_topk
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

        for offset in range(topk_group):
            group_id = (base_group + offset) % EG
            candidates = group_to_experts[group_id]

            # Get GPU loads for each expert in group
            expert_gpus = expert_to_gpu[candidates]
            loads = gpu_load[expert_gpus]

            # Get top 2 experts with lowest load
            _, indices = torch.topk(-loads, 2)  # negative for lowest
            chosen = candidates[indices]

            selected_experts.append(chosen)
            
        topk_idx[token_idx] = torch.cat(selected_experts)

    return topk_weight, topk_idx

original_biased_grouped_topk = moe_topk.biased_grouped_topk
moe_topk.biased_grouped_topk = biased_grouped_topk


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
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
):
    assert (
        routed_scaling_factor is not None
    ), "routed_scaling_factor is required for biased_grouped_topk"
    # TODO: moe_fused_gate kernel is not supported for n_share_experts_fusion > 0 now.
    if (
        _is_cuda
        and gating_output.shape[1] // num_expert_group
        <= 32  # moe_fused_gate kernel ensure that num_experts/num_expert_group does not exceed MAX_VPT=32 now. And when kernel can handle MAX_VPT > 32, we can remove this assertion.
        and is_power_of_two(correction_bias.shape[0])
    ):
        topk_weights, topk_ids = moe_fused_gate(
            gating_output,
            correction_bias,
            num_expert_group,
            topk_group,
            topk,
            n_share_experts_fusion,
            routed_scaling_factor,
        )
        # TODO merge into kernel for this branch
        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
        # TODO will fuse this into kernel, thus use slow manual operation now
        if num_token_non_padded is None:
            return topk_weights, topk_ids
        torch.compile(
            _mask_topk_ids_padded_region, dynamic=True, backend=get_compiler_backend()
        )(topk_ids, num_token_non_padded)
        return topk_weights, topk_ids
    else:
        biased_grouped_topk_fn = (
            torch.compile(
                biased_grouped_topk_impl, dynamic=True, backend=get_compiler_backend()
            )
            if compiled
            else biased_grouped_topk_impl
        )
        return biased_grouped_topk_fn(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            num_expert_group,
            topk_group,
            n_share_experts_fusion=n_share_experts_fusion,
            routed_scaling_factor=routed_scaling_factor,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

