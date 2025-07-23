import functools
import inspect
import json
import os
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.activations import ACT2FN


class TensorTracer:
    """Global tensor tracer for managing traced tensors across decorators"""

    def __init__(self, verbose: bool = False):
        self.traced_tensors = {}
        self.enabled = False
        self.verbose = verbose

    def gather_full_tensor(self, tensor, parallel_size=1, split_dim=0):
        """Gather tensor from all processes to reconstruct full tensor"""
        if not dist.is_initialized() or parallel_size == 1:
            return tensor

        # Get current process info
        world_size = dist.get_world_size()

        # Create list to hold tensors from all processes
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]

        # All-gather the tensor
        dist.all_gather(gathered_tensors, tensor)

        # Concatenate along the split dimension to reconstruct full tensor
        full_tensor = torch.cat(gathered_tensors, dim=split_dim)
        return full_tensor

    def store_tensor(self, name: str, tensor: torch.Tensor):
        """Store a traced tensor"""
        if self.enabled:
            full_tensor = self.gather_full_tensor(tensor)
            self.traced_tensors[name] = full_tensor.clone().detach().cpu()

    def _print_log(self, message: str):
        """Print log message if verbose mode is enabled"""
        if self.verbose:
            print(message)

    def enable(self):
        """Enable tensor tracing"""
        self.enabled = True
        self.traced_tensors = {}

    def disable(self):
        """Disable tensor tracing"""
        self.enabled = False

    def get_traced_tensors(self):
        """Return all traced tensors"""
        return self.traced_tensors

    def clear(self):
        """Clear all traced tensors"""
        self.traced_tensors = {}

    def save_traced_tensors(self, save_path="traced_tensors"):
        """Save traced tensors to disk"""
        os.makedirs(save_path, exist_ok=True)

        for name, tensor in self.traced_tensors.items():
            tensor_path = os.path.join(save_path, f"{name}.pt")
            torch.save(tensor, tensor_path)

        # Save metadata
        metadata = {
            "tensors": list(self.traced_tensors.keys()),
            "details": {
                name: {
                    "shape": list(tensor.shape),
                    "tp_size": tensor.shape[0] if len(tensor.shape) > 0 else 1,
                    "dtype": str(tensor.dtype),
                }
                for name, tensor in self.traced_tensors.items()
            },
        }

        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Traced tensors saved to {save_path}")


# Global tracer instance
_global_tracer = TensorTracer()


def trace_tensors(
    name: Optional[str] = None,
    trace_input: bool = True,
    trace_output: bool = True,
    tracer: Optional[TensorTracer] = None,
):
    """
    Decorator to trace input and output tensors of a function/method.

    Args:
        name: Name prefix for traced tensors. If None, uses function name.
        trace_input: Whether to trace input tensors
        trace_output: Whether to trace output tensors
        input_split_dim: Dimension along which input tensor is split in tensor parallel
        output_split_dim: Dimension along which output tensor is split in tensor parallel
        tracer: Custom tracer instance. If None, uses global tracer.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            nonlocal tracer
            if tracer is None:
                tracer = _global_tracer

            if not tracer.enabled:
                return func(self, *args, **kwargs)

            local_rank = dist.get_rank() if dist.is_initialized() else 0
            if local_rank != 0:
                # Only the first process will trace tensors
                return func(self, *args, **kwargs)

            if not hasattr(self, "__dict__"):
                # The self parameter is not an instance of a class
                raise TypeError("The first argument is assumed as 'self' ")

            # Try to get layer_id if present
            layer_id = getattr(self, "layer_id", None)
            # if layer_id is not None:
            #     if layer_id > 0:
            #         # XXX only trace the first layer
            #         return func(self, *args, **kwargs)
            # Get argument names from the original function's signature
            arg_names = inspect.getfullargspec(func).args
            # Remove 'self' if present
            if arg_names and arg_names[0] == "self":
                arg_names = arg_names[1:]

            # Determine tensor name prefix
            component_name = (
                name
                if name is not None
                else f"{self.__class__.__name__}.{func.__name__}"
            )
            if layer_id is not None:
                component_name = f"{component_name}[{layer_id}]"
            # Add a random suffix to avoid name collisions
            component_name += uuid.uuid4().hex[:4]
            # Trace input tensors

            def handle_arg(args_prefix, arg):
                tracer._print_log(
                    f"    Tracing argument: {args_prefix} of type {type(arg)} "
                )

                if isinstance(arg, torch.Tensor):
                    tracer._print_log(
                        f"      Storing tensor: {args_prefix} with shape {arg.shape} and dtype {arg.dtype}"
                    )
                    tracer.store_tensor(args_prefix, arg)
                elif isinstance(arg, (list, tuple)):
                    for i, item in enumerate(arg):
                        handle_arg(f"{args_prefix}_item{i}", item)
                elif isinstance(arg, dict):
                    for key, value in arg.items():
                        handle_arg(f"{args_prefix}_[{key}]", value)
                else:
                    tracer._print_log(
                        f"   Skipping non-tensor argument: {args_prefix} of type {type(arg)}: {arg}"
                    )

            if trace_input:
                # Trace positional arguments
                for i, arg in enumerate(args):
                    handle_arg(f"{component_name}_input_[{arg_names[i]}]", arg)
                # Trace keyword arguments
                handle_arg(f"{component_name}_input", kwargs)

            # Execute the function
            result = func(self, *args, **kwargs)

            # Trace output tensors
            if trace_output:
                handle_arg(f"{component_name}_output", result)

            return result

        return wrapper

    return decorator


@contextmanager
def tracing_enabled(tracer: Optional[TensorTracer] = None, verbose: bool = False):
    """Context manager to enable tracing for a block of code"""
    if tracer is None:
        tracer = _global_tracer

    tracer.enable()
    tracer.verbose = verbose
    try:
        yield tracer
    finally:
        tracer.disable()


def get_tracer() -> TensorTracer:
    """Get the global tracer instance"""
    return _global_tracer
