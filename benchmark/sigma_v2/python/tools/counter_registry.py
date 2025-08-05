from typing import Dict, Type
from base_counter import BaseCounter
from param_counter import ParamCounter
from mem_counter import PrefillMemCounter, DecodeMemCounter

class CounterRegistry:
    """Registry for managing different types of counters."""
    
    _counters: Dict[str, Type[BaseCounter]] = {}
    
    @classmethod
    def register(cls, name: str, counter_class: Type[BaseCounter]):
        """Register a counter class with a name."""
        cls._counters[name] = counter_class
    
    @classmethod
    def get_counter(cls, name: str, **kwargs) -> BaseCounter:
        """Get a counter instance by name."""
        if name not in cls._counters:
            raise ValueError(f"Counter '{name}' not found. Available: {list(cls._counters.keys())}")
        return cls._counters[name](**kwargs)
    
    @classmethod
    def list_counters(cls) -> list:
        """List all registered counter names."""
        return list(cls._counters.keys())
    
    @classmethod
    def get_all_counters(cls, **kwargs) -> Dict[str, BaseCounter]:
        """Get instances of all registered counters."""
        return {name: cls.get_counter(name, **kwargs) for name in cls._counters}

# Register default counters
CounterRegistry.register("param", ParamCounter)
CounterRegistry.register("prefill_mem", PrefillMemCounter)
CounterRegistry.register("decode_mem", DecodeMemCounter)