import torch
from torch import nn

from sglang.test.comparison_tests.tensor_tracer import trace_tensors
from sglang.test.comparison_tests.tf.test_mlp import MLP
from sglang.test.comparison_tests.tf.test_moe_gate import MoEGate


class MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config["num_experts_per_tok"]

        self.experts_per_rank = config["n_routed_experts"]
        self.ep_rank = 0
        self.experts = nn.ModuleList(
            [
                MLP(config, intermediate_size=config["moe_intermediate_size"])
                for _ in range(config["n_routed_experts"])
            ]
        )
        self.gate = MoEGate(config)
        if config["n_shared_experts"] is not None:
            intermediate_size = (
                config["moe_intermediate_size"] * config["n_shared_experts"]
            )
            self.shared_experts = MLP(
                config=config, intermediate_size=intermediate_size
            )

    @trace_tensors("MoE")
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config["n_shared_experts"] is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor
    ):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out
