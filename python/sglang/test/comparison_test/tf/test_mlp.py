from torch import nn
from transformers.activations import ACT2FN

from sglang.test.comparison_test.tensor_tracer import trace_tensors


class MLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = intermediate_size or config["intermediate_size"]
        act_fn = config["hidden_act"]
        self.act_fn = ACT2FN[act_fn]

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    @trace_tensors("MLP")
    def forward(self, hidden_states):
        down_proj = self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        return down_proj
