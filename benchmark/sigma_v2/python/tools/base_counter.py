from abc import ABC, abstractmethod
from model_config import ModelConfig

class BaseCounter(ABC):
    """Base class for all model component counters (params, flops, memory, etc.)"""
    
    @abstractmethod
    def embedding_lmhead(self, vocab_size: int, hidden_size: int, **kwargs) -> float:
        """Calculate metric for embedding and LM head components."""
        pass
    
    @abstractmethod
    def attention(self, 
                 hidden_size: int, 
                 num_attention_heads: int, 
                 head_dim: int,
                 num_key_value_heads: int, 
                 qk_rope_head_dim: int, 
                 qk_nope_head_dim: int, 
                 q_lora_rank: int = 0, 
                 kv_lora_rank: int = 0,
                 **kwargs) -> float:
        """Calculate metric for attention layer."""
        pass
    
    @abstractmethod
    def gateup_mlp(self, hidden_size: int, intermediate_size: int, **kwargs) -> float:
        """Calculate metric for gate-up MLP layer."""
        pass
    
    @abstractmethod
    def moe(self, 
           hidden_size: int, 
           moe_intermediate_size: int, 
           num_experts: int, 
           n_shared_experts: int = 0, 
           **kwargs) -> float:
        """Calculate metric for MoE layer."""
        pass
    
    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Return the name of the metric this counter calculates."""
        pass
    
    @property
    @abstractmethod
    def unit(self) -> float:
        """Return the unit of measurement for this metric."""
        pass
    
    def get_model_results(self,
                     seq_len: int,
                     model_config: ModelConfig) -> float:
        """Calculate the total metric for the model using the specified counter."""
        sum_count = 0
        
        # Embedding and LM head
        sum_count += self.embedding_lmhead(
            model_config.vocab_size, 
            model_config.hidden_size, 
            seq_len=seq_len
        )
        
        # Attention layers
        sum_count += self.attention(
            model_config.hidden_size,
            model_config.num_attention_heads,
            model_config.head_dim,
            model_config.num_key_value_heads,
            model_config.qk_rope_head_dim,
            model_config.qk_nope_head_dim,
            model_config.q_lora_rank,
            model_config.kv_lora_rank,
            seq_len=seq_len
        ) * model_config.total_num_layers

        # Dense MLP layers
        if model_config.total_dense_layers > 0:
            sum_count += self.gateup_mlp(
                model_config.hidden_size, 
                model_config.intermediate_size, 
                seq_len=seq_len
            ) * model_config.total_dense_layers

        # MoE layers
        if model_config.num_experts > 0:
            sum_count += self.moe(
                model_config.hidden_size,
                model_config.moe_intermediate_size,
                model_config.num_experts,
                n_shared_experts=model_config.n_shared_experts,
                seq_len=seq_len
            ) * (model_config.total_num_layers - model_config.total_dense_layers)
        
        return sum_count / self.unit

