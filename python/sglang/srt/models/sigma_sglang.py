# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference-only Sigma MoE model compatible with HuggingFace weights.

Adapted from qwen3_moe.py for the Sigma model architecture.
Key differences from Qwen3 MoE:
- Uses sigmoid scoring function for routing instead of softmax
- Uses noaux_tc topk method with e_score_correction_bias
- Has routed_scaling_factor for routing weights
- Supports shared experts (n_shared_experts)
- Uses q/k layer normalization (qk_layernorm)
- Config uses n_routed_experts instead of num_experts
"""

import logging
import math
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple, TypeVar, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding, get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.utils import (
    apply_qk_norm,
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    add_prefix,
    is_cuda,
    is_flashinfer_available,
    is_non_idle_and_non_empty,
    is_npu,
    make_layers,
)
from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
from contextlib import nullcontext

_is_cuda = is_cuda()

if _is_cuda:
    try:
        from sgl_kernel import fused_qk_norm_rope
    except ImportError:
        fused_qk_norm_rope = None

TConfig = TypeVar("TConfig", bound=PretrainedConfig)

SigmaConfig = None

_is_flashinfer_available = is_flashinfer_available()

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_npu = is_npu()

if _is_npu:
    try:
        from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope
    except ImportError:
        split_qkv_rmsnorm_rope = None


class SigmaMLP(nn.Module):
    """MLP layer for Sigma model (used for shared experts and dense layers)."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )
        return x


class SigmaTopKWithBias(nn.Module):
    """
    TopK selection for Sigma model with sigmoid scoring and correction bias.
    
    Sigma uses:
    - sigmoid scoring function instead of softmax
    - e_score_correction_bias for expert selection
    - routed_scaling_factor for final weights
    - norm_topk_prob normalization
    """
    
    def __init__(
        self,
        top_k: int,
        n_routed_experts: int,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        layer_id: int = 0,
    ):
        super().__init__()
        self.top_k = top_k
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.layer_id = layer_id
        
        # e_score_correction_bias for noaux_tc method
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(n_routed_experts)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        num_token_non_padded: Optional[int] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ):
        """
        Compute topk routing with sigmoid scoring and correction bias.
        
        Args:
            hidden_states: Input tensor of shape [num_tokens, hidden_dim]
            router_logits: Router logits of shape [num_tokens, n_routed_experts]
        
        Returns:
            TopK output with indices and weights
        """
        # Apply sigmoid scoring
        scores = router_logits.sigmoid()
        
        # Apply correction bias for noaux_tc method
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        
        # Select top-k experts
        topk_weight, topk_idx = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )
        
        # Get actual scores (without bias) for the selected experts
        topk_weight = scores.gather(1, topk_idx)
        
        # Normalize topk probabilities if configured
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        
        # Apply routed scaling factor
        topk_weight = topk_weight * self.routed_scaling_factor
        
        # Return in format compatible with sglang's MoE implementation
        # StandardTopKOutput is a NamedTuple that can be unpacked
        return StandardTopKOutput(
            topk_weights=topk_weight,
            topk_ids=topk_idx,
            router_logits=router_logits,
        )

    def empty_topk_output(self, device):
        """Return empty topk output for empty batch."""
        return StandardTopKOutput(
            topk_weights=torch.empty((0, self.top_k), dtype=torch.float32, device=device),
            topk_ids=torch.empty((0, self.top_k), dtype=torch.int64, device=device),
            router_logits=torch.empty((0, self.n_routed_experts), dtype=torch.float32, device=device),
        )


class SigmaSparseMoeBlock(nn.Module):
    """
    Sparse MoE block for Sigma model.
    
    Key differences from Qwen3MoE:
    - Uses sigmoid scoring function for routing
    - Uses noaux_tc topk method with e_score_correction_bias
    - Has routed_scaling_factor for routing weights
    - Supports shared experts (n_shared_experts)
    """
    
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        
        # Sigma uses n_routed_experts instead of num_experts
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.n_shared_experts = getattr(config, "n_shared_experts", None)
        self.moe_intermediate_size = config.moe_intermediate_size
        
        if self.tp_size > self.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.n_routed_experts}."
            )

        # Sigma uses sigmoid + bias topk selection
        self.topk = SigmaTopKWithBias(
            top_k=self.num_experts_per_tok,
            n_routed_experts=self.n_routed_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            norm_topk_prob=config.norm_topk_prob,
            layer_id=layer_id,
        )

        # Use standard MoE implementation for experts
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=self.n_routed_experts + get_global_server_args().ep_num_redundant_experts,
            top_k=self.num_experts_per_tok,
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            intermediate_size=self.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.Renormalize,
        )

        # Gate linear layer for routing
        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        # Shared experts (optional)
        if self.n_shared_experts is not None and self.n_shared_experts > 0:
            shared_intermediate_size = self.moe_intermediate_size * self.n_shared_experts
            self.shared_experts = SigmaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
            )
        else:
            self.shared_experts = None

        if get_moe_a2a_backend().is_deepep():
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                self.n_routed_experts + get_global_server_args().ep_num_redundant_experts
            )
            self.top_k = self.num_experts_per_tok

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:

        if (
            not get_moe_a2a_backend().is_deepep()
            and not get_moe_a2a_backend().is_ascend_fuseep()
        ):
            return self.forward_normal(
                hidden_states, should_allreduce_fusion, use_reduce_scatter
            )
        else:
            return self.forward_deepep(hidden_states, forward_batch)

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
        ]

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        identity = hidden_states
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_routed_experts)
        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        
        # Run through experts
        final_hidden_states = self.experts(hidden_states, topk_output)
        
        if (
            self.tp_size > 1
            and not should_allreduce_fusion
            and not use_reduce_scatter
            and not should_use_flashinfer_cutlass_moe_fp4_allgather()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        
        # Add shared experts output if exists
        if self.shared_experts is not None:
            shared_output = self.shared_experts(identity)
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states

    def forward_deepep(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        identity = hidden_states
        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_routed_experts)
            router_logits, _ = self.gate(hidden_states)
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        
        # Add shared experts output if exists
        if self.shared_experts is not None:
            shared_output = self.shared_experts(identity)
            final_hidden_states = final_hidden_states + shared_output
            
        return final_hidden_states

    def op_gate(self, state):
        if is_non_idle_and_non_empty(
            state.forward_batch.forward_mode, state.hidden_states_mlp_input
        ):
            # router_logits: (num_tokens, n_routed_experts)
            state.router_logits, _ = self.gate(state.hidden_states_mlp_input)
        else:
            state.router_logits = None

    def op_select_experts(self, state):
        router_logits = state.pop("router_logits")
        hidden_states = state.hidden_states_mlp_input
        if router_logits is not None:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.topk_output = self.topk(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    num_token_non_padded=state.forward_batch.num_token_non_padded,
                    expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    ),
                )
        else:
            state.topk_output = self.topk.empty_topk_output(hidden_states.device)

    def op_dispatch_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.dispatch_a(
                hidden_states=state.pop("hidden_states_mlp_input"),
                topk_output=state.pop("topk_output"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_dispatch_b(self, state):
        if self.ep_size > 1:
            with get_global_expert_distribution_recorder().with_current_layer(
                self.layer_id
            ):
                state.dispatch_output = self.experts.dispatcher.dispatch_b(
                    tbo_subbatch_index=state.get("tbo_subbatch_index"),
                )

    def op_experts(self, state):
        state.combine_input = self.experts.run_moe_core(
            dispatch_output=state.dispatch_output,
        )

    def op_combine_a(self, state):
        if self.ep_size > 1:
            self.experts.dispatcher.combine_a(
                combine_input=state.pop("combine_input"),
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )
            state.pop("dispatch_output")

    def op_combine_b(self, state):
        if self.ep_size > 1:
            state.hidden_states_after_combine = self.experts.dispatcher.combine_b(
                tbo_subbatch_index=state.get("tbo_subbatch_index"),
            )

    def op_output(self, state):
        state.hidden_states_mlp_output = state.pop("hidden_states_after_combine")


class SigmaAttention(nn.Module):
    """
    Multi-head attention for Sigma model.
    
    Sigma uses GQA (Grouped Query Attention) with optional QK layer normalization.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        qk_layernorm: bool = True,
        config: Optional[TConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.config = config
        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()
        self.qk_layernorm = qk_layernorm

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        # Check if rotary_emb uses fallback kernel - if so, fused kv buffer is not supported
        rotary_uses_fallback = getattr(self.rotary_emb, 'use_fallback_kernel', False)
        self.compatible_with_fused_kv_buffer = (
            (not isinstance(self.rotary_emb, MRotaryEmbedding)) 
            and (not rotary_uses_fallback)
        )
        self.compatible_with_fused_qk_norm_rope = (
            not isinstance(self.rotary_emb, MRotaryEmbedding)
        ) and self.head_dim in (64, 128, 256) and self.qk_layernorm
        self.use_fused_qk_norm_rope = (
            get_global_server_args().enable_fused_qk_norm_rope
            and self.compatible_with_fused_qk_norm_rope
            and fused_qk_norm_rope is not None
        )
        self._used_fused_qk_norm_rope_last_call = False

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

        # QK layer normalization (Sigma specific)
        if self.qk_layernorm:
            self.q_layernorm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_layernorm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        else:
            self.q_layernorm = None
            self.k_layernorm = None
        
        self.alt_stream = alt_stream

    def op_prepare(self, state):
        state.attn_intermediate_state = self.forward_prepare(
            positions=state.positions,
            hidden_states=state.pop("hidden_states_after_comm_pre_attn"),
            forward_batch=state.forward_batch,
        )

    def op_core(self, state):
        state.hidden_states_after_attn = self.forward_core(
            state.pop("attn_intermediate_state")
        )

    def forward_prepare_npu(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        if self.qk_layernorm and split_qkv_rmsnorm_rope is not None:
            if self.attn.layer_id == forward_batch.token_to_kv_pool.start_layer:
                self.rotary_emb.get_cos_sin_with_position(positions)
            q, k, v = split_qkv_rmsnorm_rope(
                qkv,
                self.rotary_emb.position_sin,
                self.rotary_emb.position_cos,
                self.q_layernorm.weight,
                self.k_layernorm.weight,
                self.q_size,
                self.kv_size,
                self.head_dim,
                self.q_layernorm.variance_epsilon,
                q_bias=getattr(self.q_layernorm, "bias", None),
                k_bias=getattr(self.k_layernorm, "bias", None),
            )
        else:
            q, k, v = self.apply_qk_norm_rope(qkv, positions, forward_batch)

        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state

    def forward_prepare_native(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = self.apply_qk_norm_rope(qkv, positions, forward_batch)
        inner_state = q, k, v, forward_batch
        return None, forward_batch, inner_state

    def apply_qk_norm_rope(self, qkv, positions, forward_batch):
        use_fused = (
            self.use_fused_qk_norm_rope 
            and qkv.dtype == torch.bfloat16 
            and self.qk_layernorm
        )
        if use_fused and fused_qk_norm_rope is not None:
            theta = getattr(self.config, "rope_theta", 10000.0)
            positions = (
                positions.view(-1).to(dtype=torch.int32, device=qkv.device).contiguous()
            )
            # Sigma doesn't use yarn rope scaling by default
            factor, low, high, attention_factor = 1.0, 0, 0, 1.0
            fused_qk_norm_rope(
                qkv,
                self.num_heads,
                self.num_kv_heads,
                self.num_kv_heads,
                self.head_dim,
                self.q_layernorm.variance_epsilon,
                self.q_layernorm.weight,
                self.k_layernorm.weight,
                theta,
                self.rotary_emb.is_neox_style,
                positions,
                factor,
                low,
                high,
                attention_factor,
            )
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            self._used_fused_qk_norm_rope_last_call = True
        else:
            # Fallback to non-fused implementation
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            
            # Apply QK layer normalization if enabled
            if self.qk_layernorm:
                q, k = apply_qk_norm(
                    q=q,
                    k=k,
                    q_norm=self.q_layernorm,
                    k_norm=self.k_layernorm,
                    head_dim=self.head_dim,
                    alt_stream=self.alt_stream,
                )
            
            q, k = self.rotary_emb(
                positions,
                q,
                k,
                fused_set_kv_buffer_arg=(
                    create_fused_set_kv_buffer_arg(
                        value=v,
                        layer=self.attn,
                        forward_batch=forward_batch,
                    )
                    if enable_fused_set_kv_buffer(forward_batch)
                    and self.compatible_with_fused_kv_buffer
                    else None
                ),
            )
            self._used_fused_qk_norm_rope_last_call = False
        return q, k, v

    def forward_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, forward_batch, None
        if not _is_npu:
            return self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        else:
            return self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

    def forward_core(self, intermediate_state):
        hidden_states, forward_batch, inner_state = intermediate_state
        if inner_state is None:
            return hidden_states

        q, k, v, fb = inner_state

        must_save_kv = self._used_fused_qk_norm_rope_last_call
        save_kv_cache = must_save_kv or not (
            enable_fused_set_kv_buffer(forward_batch)
            and self.compatible_with_fused_kv_buffer
        )
        attn_output = self.attn(
            q,
            k,
            v,
            fb,
            save_kv_cache=save_kv_cache,
        )
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        s = self.forward_prepare(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        return self.forward_core(s)


class SigmaDecoderLayer(nn.Module):
    """Decoder layer for Sigma model."""
    
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rms_norm_eps = config.rms_norm_eps
        attention_bias = getattr(config, "attention_bias", False)
        qk_layernorm = getattr(config, "qk_layernorm", True)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        
        self.self_attn = SigmaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            qk_layernorm=qk_layernorm,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            dual_chunk_attention_config=dual_chunk_attention_config,
            alt_stream=alt_stream,
        )

        self.layer_id = layer_id

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        # Sigma: all layers after first_k_dense_replace are sparse MoE layers
        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        self.is_layer_sparse = (
            config.n_routed_experts is not None 
            and layer_id >= first_k_dense_replace
        )
        is_previous_layer_sparse = (
            config.n_routed_experts is not None 
            and (layer_id - 1) >= first_k_dense_replace
        ) if layer_id > 0 else False
        is_next_layer_sparse = (
            config.n_routed_experts is not None 
            and (layer_id + 1) >= first_k_dense_replace
        ) if layer_id < config.num_hidden_layers - 1 else False

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = SigmaSparseMoeBlock(
                layer_id=self.layer_id,
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = SigmaMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size if hasattr(config, 'intermediate_size') else config.moe_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(self.layer_id == self.config.num_hidden_layers - 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
            )
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        hidden_states = self.mlp(
            hidden_states, forward_batch, should_allreduce_fusion, use_reduce_scatter
        )

        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual

    def op_comm_prepare_attn(
        self,
        state,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        tbo_subbatch_index: Optional[int] = None,
    ):
        state.hidden_states_after_comm_pre_attn, state.residual_after_input_ln = (
            self.layer_communicator.prepare_attn(hidden_states, residual, forward_batch)
        )
        state.update(
            dict(
                forward_batch=forward_batch,
                positions=positions,
                tbo_subbatch_index=tbo_subbatch_index,
            )
        )

    def op_comm_prepare_mlp(self, state):
        state.hidden_states_mlp_input, state.residual_after_comm_pre_mlp = (
            self.layer_communicator.prepare_mlp(
                state.pop("hidden_states_after_attn"),
                state.pop("residual_after_input_ln"),
                state.forward_batch,
            )
        )

    def op_mlp(self, state):
        hidden_states = state.pop("hidden_states_mlp_input")
        state.hidden_states_mlp_output = self.mlp(hidden_states, state.forward_batch)

    def op_comm_postprocess_layer(self, state):
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            state.pop("hidden_states_mlp_output"),
            state.pop("residual_after_comm_pre_mlp"),
            state.forward_batch,
        )

        output = dict(
            positions=state.positions,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=state.forward_batch,
            tbo_subbatch_index=state.tbo_subbatch_index,
        )

        state.clear(
            expect_keys={
                "positions",
                "forward_batch",
                "tbo_subbatch_index",
            }
        )
        return output


class SigmaModel(nn.Module):
    """Sigma Model backbone."""
    
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to SigmaDecoderLayer
        decoder_layer_type = decoder_layer_type or SigmaDecoderLayer
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

    def set_eagle3_layers_to_capture(self, layers_to_capture: List[int]):
        self.layers_to_capture = layers_to_capture
        for layer_id in self.layers_to_capture:
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        if forward_batch.can_run_tbo:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers,
                enable_tbo=True,
                input_data_scatter_mode=ScatterMode.model_input_output(),
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:
            for i in range(self.start_layer, self.end_layer):
                ctx = (
                    nullcontext()
                    if get_global_server_args().enable_piecewise_cuda_graph
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    layer = self.layers[i]
                    hidden_states, residual = layer(
                        positions,
                        hidden_states,
                        forward_batch,
                        residual,
                        captured_last_layer_outputs=(
                            aux_hidden_states
                            if getattr(layer, "_is_layer_to_capture", False)
                            else None
                        ),
                    )
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class SigmaForCausalLM(nn.Module):
    """Sigma model for causal language modeling."""
    
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.model = SigmaModel(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
            alt_stream=alt_stream,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds

        # decoder layer
        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                forward_batch.hidden_states, forward_batch.residual = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.set_eagle3_layers_to_capture(
                [
                    2,
                    num_layers // 2,
                    num_layers - 3,
                ]
            )  # Specific layers for EAGLE3 support
        else:
            self.model.set_eagle3_layers_to_capture([val + 1 for val in layer_ids])

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load model weights with Sigma-specific mappings."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Sigma uses n_routed_experts instead of num_experts
        n_routed_experts = self.config.n_routed_experts

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=n_routed_experts,
        )

        # Cache params_dict to avoid repeated expensive traversal of model parameters
        if not hasattr(self, "_cached_params_dict"):
            self._cached_params_dict = dict(self.named_parameters())
        params_dict = self._cached_params_dict
        
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name:
                continue
            
            # Handle Sigma-specific weight name mappings
            # Map q_layernorm/k_layernorm to our naming
            if "q_layernorm" in name or "k_layernorm" in name:
                # Already matches our naming
                pass
            
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Mark as expert weight regardless of whether we can process it
                    is_expert_weight = True

                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        # Expert weight not on this rank, will be skipped below
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

        # Lazy initialization of expert weights cache to avoid slowing down load_weights
        if not hasattr(self, "routed_experts_weights_of_layer"):
            self.routed_experts_weights_of_layer = {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.start_layer, self.end_layer)
                if isinstance(self.model.layers[layer_id].mlp, SigmaSparseMoeBlock)
            }

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )

class SigmaV2ForCausalLM(SigmaForCausalLM):
    pass

# Entry class for sglang model registry
EntryClass = [SigmaForCausalLM, SigmaV2ForCausalLM]
