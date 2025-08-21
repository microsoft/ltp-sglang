import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class MockLogitsConfig:
    """
    Mock configuration class for testing purposes.
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class LogitsModule(nn.Module):
    """
    Logits Module for testing
    """

    def __init__(self, config):
        super().__init__()
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
        )
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self, hidden_states: torch.Tensor, logits_metadata: LogitsMetadata
    ) -> torch.Tensor:
        """Forward pass for the logits module."""
        return self.logits_processor(
            None,  # No input_ids needed for logits processing
            hidden_states,
            self.lm_head,
            logits_metadata,
        )
