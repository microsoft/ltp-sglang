import os
from typing import Any, Dict, Iterable, Optional, Tuple, Type

import torch
from torch import nn

from sglang import ServerArgs
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear

# from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_loader.weight_utils import default_weight_loader


class AttentionLayer(nn.Module):
    """
    Attention Layer with Rotary Embedding and Radix Attention
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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= self.tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.qk_layernorm = qk_layernorm
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

        self.q_layernorm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_by_head = q.reshape(-1, self.head_dim)
        q_by_head = self.q_layernorm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_layernorm(k_by_head)
        k = k_by_head.view(k.shape)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.qk_layernorm:
            # Apply layer normalization to q and k
            q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
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
                # If no matching stacked parameter found, load directly
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


class MockModelRunner:
    def __init__(
        self,
        page_size=1,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float16,
        max_batch_size=1,
        max_context_len=4096,
    ):
        self.device = "cuda"
        self.gpu_id = 0
        self.dtype = dtype
        attention_arch = AttentionArch.MHA

        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": max_context_len,
                "is_multimodal": False,
                "attention_arch": attention_arch,
                "get_num_kv_heads": lambda _: num_kv_heads,
            },
        )
        self.sliding_window_size = None
        self.device = self.device
        # Create a large enough req_to_token_pool to fit the test usage.
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                # A typical max_bs * max_context_len for cuda graph decode
                "size": max_batch_size,
                # Add req_to_token attribute
                "req_to_token": torch.zeros(
                    max_batch_size,
                    max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            },
        )
        self.page_size = page_size
        max_total_num_tokens = max_batch_size * max_context_len
        self.token_to_kv_pool = MHATokenToKVPool(
            size=max_total_num_tokens,
            page_size=page_size,
            dtype=self.dtype,
            head_num=num_kv_heads,
            head_dim=head_dim,
            layer_num=1,  # only consider layer=1 for unit test
            device=self.device,
            enable_memory_saver=False,
        )
        # Required by torch native backend
        self.server_args = ServerArgs(model_path="fake_model_path")


class AttentionLayerTester:
    """Wrapper for testing AttentionLayer with different backends."""

    def __init__(
        self,
        attention_layer: AttentionLayer,
        attn_backend: Type[AttentionBackend],
        batch_size: int,
        seq_len: int,
    ):
        """Initialize the AttentionLayerTester with an attention layer and backend."""
        self.attention_layer = attention_layer
        self.num_heads = attention_layer.num_heads
        self.head_dim = attention_layer.head_dim
        self.dtype = attention_layer.qkv_proj.weight.dtype
        self.device = attention_layer.qkv_proj.weight.device

        self.mock_runner = MockModelRunner(
            num_kv_heads=attention_layer.num_kv_heads,
            head_dim=attention_layer.head_dim,
            dtype=self.dtype,
            max_batch_size=batch_size,
            max_context_len=seq_len,
        )
        self.mock_runner.model_config.num_attention_heads = attention_layer.num_heads
        self.attn_backend = attn_backend(self.mock_runner)

        self.batch_size = batch_size
        self.seq_len = seq_len

    def _mock_write_to_req_to_token_pool(self, batch_size, seq_len, page_size=1):
        # if page_size > 1, the token pool stores the index to the page.
        # so we need to multiply the index by page_size.
        self.req_to_token = (
            torch.arange(0, batch_size, dtype=torch.int32, device=self.device)[:, None]
            * seq_len
            + torch.arange(0, seq_len, dtype=torch.int32, device=self.device)[None, :]
            + page_size
        )
        self.mock_runner.req_to_token_pool.req_to_token[:batch_size, :seq_len] = (
            self.req_to_token
        )

    def _create_forward_batch(self, q_len=None, prefix_len=0):
        """Create a forward batch for testing based on mode and lengths."""
        # Default to self.seq_len if not specified
        q_len = q_len or self.seq_len

        total_len = prefix_len + q_len
        out_cache_start = prefix_len * self.batch_size
        out_cache_end = total_len * self.batch_size

        forward_batch = ForwardBatch(
            batch_size=self.batch_size,
            input_ids=torch.randint(
                0, 100, (self.batch_size, q_len), device=self.device
            ),
            out_cache_loc=torch.arange(
                out_cache_start, out_cache_end, device=self.device
            ),
            seq_lens_sum=self.batch_size * total_len,
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=torch.arange(self.batch_size, device=self.device),
            seq_lens=torch.tensor([total_len] * self.batch_size, device=self.device),
            seq_lens_cpu=torch.tensor([total_len] * self.batch_size, device="cpu"),
            extend_prefix_lens=torch.tensor(
                [prefix_len] * self.batch_size, device=self.device
            ),
            extend_prefix_lens_cpu=torch.tensor(
                [prefix_len] * self.batch_size, device="cpu"
            ),
            extend_seq_lens=torch.tensor([q_len] * self.batch_size, device=self.device),
            extend_seq_lens_cpu=torch.tensor([q_len] * self.batch_size, device="cpu"),
            attn_backend=self.attn_backend,
        )

        # Add token pool
        forward_batch.req_to_token_pool = self.mock_runner.req_to_token_pool

        # Write current batch's req_to_token to req_to_token_pool
        self._mock_write_to_req_to_token_pool(self.batch_size, total_len)
        # Add kv pool for this forward batch
        forward_batch.token_to_kv_pool = self.mock_runner.token_to_kv_pool

        return forward_batch

    def _setup_kv_cache(self, forward_batch, cache_len):
        # Create constant values for the prefix cache for easy debugging
        cache_k = torch.ones(
            self.batch_size * cache_len,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        cache_v = (
            torch.ones(
                self.batch_size * cache_len,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            * 2
        )

        # Set the prefix KV cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attention_layer.attn,
            torch.arange(self.batch_size * cache_len, device=self.device),
            cache_k,
            cache_v,
            self.attention_layer.attn.k_scale,
            self.attention_layer.attn.v_scale,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the attention layer."""
        forward_batch = self._create_forward_batch(q_len=self.seq_len)
        self.attn_backend.init_forward_metadata(forward_batch)

        # XXX
        # self._setup_kv_cache(forward_batch, cache_len=self.seq_len)
        return self.attention_layer(positions, hidden_states, forward_batch)
