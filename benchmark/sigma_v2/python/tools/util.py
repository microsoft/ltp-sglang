from typing import Dict, Any

def get_config_value(config: Dict[str, Any], keys: list, default: Any = None) -> Any:
    """Retrieve a value from config using multiple possible keys."""
    for key in keys:
        if key in config:
            return config[key]
    return default