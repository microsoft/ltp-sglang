import pytest
import torch

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.test.numerical_tests.common import *
from sglang.test.numerical_tests.modules.test_attention import (
    AttentionLayer,
    AttentionLayerTester,
)
from sglang.test.numerical_tests.test_module import TestModule
from sglang.test.numerical_tests.utils.load_data import (
    load_random_weights,
    load_weight_from_hf_ckp,
)

# Define the configurations for the Attention Layer tests
AttentionLayer_configs = [
    {
        "hidden_size": 5120,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "rope_theta": 10000,
        "rope_scaling": {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
        "max_position_embeddings": 4096,
        "head_dim": 128,
        "rms_norm_eps": 1e-06,
        "attention_bias": False,
        "qk_layernorm": True,
    }
]
# Real weight prefixes in the model checkpoint for the Attention Layer tests
weight_prefixes = [
    "model.layers.0.self_attn",
    "model.layers.20.self_attn",
    "model.layers.40.self_attn",
    "model.layers.62.self_attn",
    "random0",
    "random1",
    "random2",
    "random3",
]
attention_backends = [
    TritonAttnBackend,
    TorchNativeAttnBackend,
    # FlashAttentionBackend,
]


class TestAttentionLayer(TestModule):
    """
    Test the consistency of Attention Layer computation.
    """

    def setup_method(self, method):
        super().setup_method(method)
        # Initialize the distributed environment for Triton attention backend
        initialize_dp_attention(
            enable_dp_attention=False,
            tp_rank=0,
            tp_size=1,
            dp_size=1,
            moe_dense_tp_size=None,
            pp_size=1,
        )

    @pytest.mark.parametrize("module_config", AttentionLayer_configs)
    @pytest.mark.parametrize("weight_prefix", weight_prefixes)
    @pytest.mark.parametrize("attn_backend", attention_backends)
    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_sglang_attention_layer(
        self, module_config, weight_prefix, attn_backend, dtype
    ):
        """Test the Attention Layer with random weights."""
        # Create the Attention Layer with the given configuration
        sgl_module = AttentionLayer(
            hidden_size=module_config["hidden_size"],
            num_heads=module_config["num_attention_heads"],
            num_kv_heads=module_config["num_key_value_heads"],
            rope_theta=module_config["rope_theta"],
            rope_scaling=module_config["rope_scaling"],
            max_position_embeddings=module_config["max_position_embeddings"],
            head_dim=module_config["head_dim"],
            rms_norm_eps=module_config["rms_norm_eps"],
            attention_bias=module_config["attention_bias"],
            qk_layernorm=module_config["qk_layernorm"],
        )
        if weight_prefix.count("random") > 0:
            # Load random weights if the prefix indicates so
            load_random_weights(sgl_module, dtype=dtype)
        else:
            # Load weights from the Hugging Face checkpoint
            load_weight_from_hf_ckp(sgl_module, weight_prefix, dtype=dtype)

        log_dir = os.path.join(
            LOG_DIR,
            "attention_layer",
            f"{module_config['hidden_size']}-{module_config['num_attention_heads']}-{module_config['num_key_value_heads']}-{dtype}",
            f"weights-{weight_prefix}",
        )
        os.makedirs(log_dir, exist_ok=True)

        def forward_func(inputs) -> torch.Tensor:
            """Forward function for the Attention Layer."""
            position, input_tensor, batch_size, seq_len = inputs
            # Create a forward batch with the required metadata
            module_tester = AttentionLayerTester(
                sgl_module,
                attn_backend=attn_backend,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            # Clone the input tensor to avoid in-place operations
            input_tensor_clone = input_tensor.clone()

            return module_tester.forward(position, input_tensor_clone)

        def random_input_func(bs: int, sl: int, dtype: torch.dtype) -> torch.Tensor:
            """Generate a random input tensor for the Attention Layer."""
            hidden_states = torch.randn(
                bs * sl, module_config["hidden_size"], dtype=dtype
            ).cuda()
            positions = (
                torch.arange(sl, dtype=torch.int64, device=hidden_states.device)
                .repeat(bs, 1)
                .unsqueeze(0)
            )
            return positions, hidden_states, bs, sl

        self._run_module_random_input(
            forward_func, random_input_func, dtype, log_dir=log_dir
        )
