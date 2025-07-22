import torch
import sglang.srt.layers.moe.topk as moe_topk
from sgl_kernel import topk_softmax

import triton
import triton.language as tl
import torch

@triton.jit
def balanced_topk_kernel_variable_capacity(
    gating_output_ptr,       # float32[M, N]
    token_counts_ptr,        # int32[M]
    expert_counts_ptr,       # int32[N]
    max_expert_counts_ptr,   # int32[N]
    topk_ids_ptr,            # int32[M, topk]
    token_stride,            # int32
    expert_stride,           # int32
    M: tl.constexpr,
    N: tl.constexpr,
    TOPK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid // N  # token index
    j = pid % N   # expert index

    if i >= M:
        return

    offset = i * expert_stride + j
    score = tl.load(gating_output_ptr + offset)

    tok_cnt = tl.load(token_counts_ptr + i)
    exp_cnt = tl.load(expert_counts_ptr + j)
    exp_max = tl.load(max_expert_counts_ptr + j)

    if tok_cnt < TOPK and exp_cnt < exp_max:
        new_tok_cnt = tl.atomic_add(token_counts_ptr + i, 1)
        new_exp_cnt = tl.atomic_add(expert_counts_ptr + j, 1)

        if new_tok_cnt < TOPK:
            out_offset = i * TOPK + new_tok_cnt
            tl.store(topk_ids_ptr + out_offset, j)
        else:
            tl.atomic_add(token_counts_ptr + i, -1)
            tl.atomic_add(expert_counts_ptr + j, -1)

def balanced_topk_triton(gating_output: torch.Tensor, topk: int):
    M, N = gating_output.shape
    device = gating_output.device

    total_assignments = M * topk
    base = total_assignments // N
    rem = total_assignments % N

    # Expert capacity: first 'rem' get +1
    max_expert_counts = torch.full((N,), base, dtype=torch.int32, device=device)
    if rem > 0:
        max_expert_counts[:rem] += 1

    # Buffers
    topk_ids = torch.full((M, topk), -1, dtype=torch.int32, device=device)
    token_counts = torch.zeros(M, dtype=torch.int32, device=device)
    expert_counts = torch.zeros(N, dtype=torch.int32, device=device)

    grid = lambda meta: (M * N,)

    # Triton launch
    balanced_topk_kernel_variable_capacity[grid](
        gating_output,
        token_counts,
        expert_counts,
        max_expert_counts,
        topk_ids,
        gating_output.stride(0),
        gating_output.stride(1),
        M=M,
        N=N,
        TOPK=topk,
    )

    return topk_ids

def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    token_expert_indicies = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),
    )
    del token_expert_indicies

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    topk_ids = balanced_topk_triton(gating_output, topk)
    return topk_weights, topk_ids

original_fused_topk = moe_topk.fused_topk
moe_topk.fused_topk = fused_topk
