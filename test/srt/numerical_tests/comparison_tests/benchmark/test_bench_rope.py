import os

import pytest

from sglang.test.numerical_tests.bench_module import BenchConfig, BenchModule
from sglang.test.numerical_tests.modules.tf.test_rope import RotaryEmbedding
from sglang.test.numerical_tests.utils.common import *

# Define the configurations for the Rotary Embedding tests
ROPE_configs = [
    {
        "head_dim": 128,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "max_position_embeddings": 4096,
        "rope_theta": 10000,
        "rope_scaling": None,
        "hidden_size": 5120,
    }
]


class TestBenchRoPE(BenchModule):
    """
    Test the Rotary Position Embedding Layer computation with a benchmark (Transformers' implementation).
    """

    @pytest.mark.parametrize("rope_config", ROPE_configs)
    def test_bench_rope(self, rope_config):
        """Run the sglang RoPE with random input and trace tensors."""
        # Initialize the Rotary Embedding module with the given configurations
        rope_module = RotaryEmbedding(rope_config)
        rope_module.to(TARGET_DTYPE).cuda()

        # Create a directory for tracing tensors
        log_dir = os.path.join(
            LOG_DIR,
            "bench",
            "rope",
            f"{rope_config['head_dim']}_{rope_config['num_attention_heads']}",
            f"rope",
        )

        # Define the random input function for the Rotary Position Embedding layer
        def forward_func(inputs) -> torch.Tensor:
            """Forward function for the Rotary Position Embedding layer."""
            query = inputs["query"]
            key = inputs["key"]
            position_ids = torch.arange(
                0, query.shape[2], dtype=torch.int64, device=query.device
            ).unsqueeze(0)
            # Only the seq len of hidden_states used by the RoPE module
            hidden_states = torch.tensor(
                (query.shape[0], query.shape[2], rope_config["hidden_size"]),
                dtype=query.dtype,
                device=query.device,
            )
            q_embed, k_embed = rope_module(position_ids, query, key, hidden_states)
            res = torch.cat((q_embed, k_embed), dim=1)

            return res

        # Define the random input function for the Rotary Position Embedding layer
        def random_input_func(batch_size, seq_len, dtype=TARGET_DTYPE):
            """Generate random input tensor for the Rotary Position Embedding layer."""
            query = torch.randn(
                batch_size,
                rope_config["num_attention_heads"],
                seq_len,
                rope_config["head_dim"],
                dtype=dtype,
            ).cuda()
            key = torch.randn(
                batch_size,
                rope_config["num_key_value_heads"],
                seq_len,
                rope_config["head_dim"],
                dtype=dtype,
            ).cuda()
            return {"query": query, "key": key}

        # Create a BenchConfig instance for the RoPE module
        bench_config = BenchConfig(
            module_config=rope_config,
            real_weight_prefix="",
            log_dir=log_dir,
            batch_sizes=BATCH_SIZES,
            seq_lens=SEQ_LENS,
            dtype=TARGET_DTYPE,
            input_count=RANDOM_INPUT_COUNT,
            repeat_count=REPEAT_COUNT,
        )

        # Run the RoPE module with random input and trace tensors
        self._run_module_random_input(
            rope_module,
            bench_config,
            forward_func,
            random_input_func,
            log_dir,
        )
