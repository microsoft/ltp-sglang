import json
from dataclasses import dataclass, fields, MISSING
from typing import Optional, Dict, Any

PRECISION_TO_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp8": 1
}

SUPPORTED_MODELS = ['qwen3_moe', 'deepseek_v3', 'sigma']

@dataclass
class ModelConfig:
    name: str
    total_num_layers: int
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    
    precision: Optional[str] = "fp16"
    precision_bytes: Optional[int] = 2
    
    quantization_block_size: Optional[int] = 1
    
    num_key_value_heads: Optional[int] = None
    total_dense_layers: Optional[int] = 0
    
    n_shared_experts: Optional[int] = 0
    
    q_lora_rank: Optional[int] = 0
    kv_lora_rank: Optional[int] = 0
    qk_rope_head_dim: Optional[int] = 0
    qk_nope_head_dim: Optional[int] = 0
    
    
    FIELD_ALIASES = {
        "name": ["model_type"],
        "total_num_layers": ["num_hidden_layers", "n_layer"],
        "total_dense_layers": ["first_k_dense_replace"],
        "num_attention_heads": ["num_attention_heads", "n_head"],
        "hidden_size": ["hidden_size", "n_embed"],
        "intermediate_size": ["intermediate_size", "n_inner"],
        "num_experts": ["n_routed_experts", "num_local_experts", "num_experts"],
        "head_dim": ["head_dim", "v_head_dim"],
    }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        result = {}
        missing_required = []
        for f in fields(cls):
            field_name = f.name
            if field_name in config_dict:
                result[field_name] = config_dict[field_name]
                continue
            aliases = cls.FIELD_ALIASES.get(field_name, [])
            for alias in aliases:
                if alias in config_dict:
                    result[field_name] = config_dict[alias]
                    break
            else:
                if f.default is MISSING and f.default_factory is MISSING:
                    missing_required.append(field_name)
        if  'quantization_config' in config_dict:
            quant_method = config_dict['quantization_config'].get('quant_method', None)
            result['precision'] = quant_method if quant_method else "bf16"
        
        if missing_required:
            raise ValueError(f"Missing required fields in config: {', '.join(missing_required)}")
        if not result['name'] in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {result['name']}. Supported models are: {', '.join(SUPPORTED_MODELS)}")
        return cls(**result)
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'ModelConfig':
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        obj = cls.from_dict(config_dict)
        return obj