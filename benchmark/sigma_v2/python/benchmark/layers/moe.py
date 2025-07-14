import os
import ctypes
import argparse
import json
from typing import Any, Callable, Dict, List, Tuple

import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from triton.testing import do_bench

from layer import LayerBenchmark
from constants import layer_configs

# Load libnuma and set function types
libnuma = ctypes.CDLL("libnuma.so")
libnuma.numa_available.restype = ctypes.c_int
libnuma.numa_set_preferred.argtypes = [ctypes.c_int]


def set_numa_affinity(rank: int) -> None:
    """Set NUMA affinity and CPU affinity for the current process based on rank."""
    cores_per_rank = 12
    numa_node = rank // 4
    core_start = rank * cores_per_rank
    core_end = core_start + cores_per_rank
    p = psutil.Process(os.getpid())
    p.cpu_affinity(list(range(core_start, core_end)))
    
    if libnuma.numa_available() != -1:
        libnuma.numa_set_preferred(numa_node)


def init_dist(local_rank: int, num_local_ranks: int) -> Tuple[int, int, torch.distributed.ProcessGroup]:
    """
    Initialize distributed process group with NCCL backend and set device.
    
    Args:
        local_rank: Local GPU index.
        num_local_ranks: Number of GPUs per node.
        
    Returns:
        A tuple containing (rank, world_size, process_group).
    """
    ip = os.getenv("MASTER_ADDR", "10.0.0.10")
    port = int(os.getenv("MASTER_PORT", "30000"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))
    
    # If multiple nodes, ensure conditions on local ranks are met.
    assert (num_local_ranks < 8 and num_nodes == 1) or num_local_ranks == 8, "Invalid configuration."

    world_size = num_nodes * num_local_ranks
    rank = node_rank * num_local_ranks + local_rank

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{ip}:{port}",
        world_size=world_size,
        rank=rank,
    )
    set_numa_affinity(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    group = dist.new_group(list(range(world_size)))
    return dist.get_rank(), world_size, group


def bench(fn: Callable[[], Any], num_warmups: int = 20, num_tests: int = 30,
          post_fn: Callable[[], Any] = None) -> float:
    """
    Benchmark a given function using CUDA events.

    Args:
        fn: The function to benchmark.
        num_warmups: Number of warmup iterations.
        num_tests: Number of timed iterations.
        post_fn: Optional function to run after each iteration.

    Returns:
        Median execution time in seconds.
    """
    torch.cuda.synchronize()
    # Allocate an array to clear cache; size in integers (each 4 bytes)
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    for _ in range(num_warmups):
        fn()

    cache.zero_()

    # Create CUDA events for timing
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_tests)]
    for i in range(num_tests):
        start_events[i].record()
        fn()
        end_events[i].record()
        if post_fn is not None:
            post_fn()
    torch.cuda.synchronize()

    # Convert elapsed time from milliseconds to seconds and skip the first measurement.
    times = np.array([s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)])[1:]
    return float(np.median(times))


class MoEGate(nn.Module):
    """
    A simple gate used in Mixture of Experts (MoE) layer.
    """
    def __init__(self, num_experts: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((num_experts, hidden_size), dtype=torch.bfloat16)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((num_experts))
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Linear projection: (bsz, hidden_size) x (hidden_size, num_experts)^T
        logits = F.linear(hidden_states, self.weight, None)
        return logits


class MoEBenchmark(LayerBenchmark):
    """
    Benchmarking class for various Mixture of Experts implementations.
    """
    def __init__(
        self,
        local_rank: int = 0,
        mp_size: int = 8,
        routed_scaling_factor: float = 1.0,
        num_experts: int = 256,
        hidden_size: int = 7168,
        intermediate_size: int = 2048,
        top_k: int = 8,
        use_grouped_topk: bool = True,
        num_expert_group: int = 8,
        topk_group: int = 4,
        renormalize: bool = True,
    ) -> None:
        super().__init__("MoE")
        self.mp_size = mp_size
        self.device = torch.device(f"cuda:{local_rank}")
        self.routed_scaling_factor = routed_scaling_factor
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.renormalize = renormalize

        # Initialize gate model and move to device
        self.gate = MoEGate(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size
        ).to(self.device)
        self.correction_bias = self.gate.e_score_correction_bias.data

        # Apply parallel patch
        import mp_patch
        mp_patch.apply_hardcode_parallel(mp_size)

        from sglang.srt.layers.quantization.fp8 import Fp8Config
        self.quant_config = Fp8Config.from_config({
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128]
        })

        # Mapping benchmark function names to methods
        self.benchmark_funcs: Dict[str, Callable[..., Tuple[float, int, int]]] = {
            "fused_moe": self.bench_fused_moe,
            "ep_moe": self.bench_ep_moe,
            "normal_deepep_moe": self.bench_normal_deepep_moe,
            "low_lat_deepep_moe": self.bench_low_lat_deepep_moe,
        }

    def bench_fused_moe(self, bsz: int, seq_len: int,
                        process_group: torch.distributed.ProcessGroup = None) -> Tuple[float, int, int]:
        """
        Benchmark fused MoE implementation.
        """
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.utils import add_prefix

        experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            renormalize=self.renormalize,
            quant_config=self.quant_config,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            correction_bias=None,
            tp_size=self.mp_size,
            params_dtype=torch.half,
            prefix=add_prefix("experts", prefix=""),
        ).to(self.device)

        hidden_states = torch.randn(
            bsz * seq_len, self.hidden_size, device=self.device, dtype=torch.bfloat16
        )

        def run_bench() -> torch.Tensor:
            router_logits = self.gate(hidden_states)
            return experts(hidden_states=hidden_states, router_logits=router_logits)

        ms = bench(run_bench)
        return ms, 0, 0

    def bench_ep_moe(self, bsz: int, seq_len: int,
                     process_group: torch.distributed.ProcessGroup) -> Tuple[float, int, int]:
        """
        Benchmark expert parallel MoE (EPMoE).
        """
        from sglang.srt.layers.moe.ep_moe.layer import EPMoE
        
        experts = EPMoE(
            layer_id=0,
            num_experts=self.num_experts,
            top_k=self.top_k + 1,  
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            renormalize=self.renormalize,
            quant_config=self.quant_config,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            tp_size=self.mp_size,
        ).to(self.device)
        
        hidden_states = torch.randn(
            bsz * seq_len, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )

        def run_bench() -> torch.Tensor:
            router_logits = self.gate(hidden_states)
            return experts(hidden_states=hidden_states, router_logits=router_logits)

        ms = bench(run_bench)
        return ms, 0, 0

    def bench_normal_deepep_moe(self, bsz: int, seq_len: int,
                                process_group: torch.distributed.ProcessGroup) -> Tuple[float, int, int]:
        """
        Benchmark the normal deepep MoE implementation.
        """
        from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
        from sglang.srt.utils import DeepEPMode
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.layers.moe.topk import select_experts
        from sglang.srt.layers.moe.ep_moe.token_dispatcher import DeepEPDispatcher

        experts = DeepEPMoE(
            layer_id=0,
            num_experts=self.num_experts,
            top_k=self.top_k + 1,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            renormalize=self.renormalize,
            quant_config=self.quant_config,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            tp_size=self.mp_size,
            correction_bias=self.gate.e_score_correction_bias,
            deepep_mode=DeepEPMode.normal
        ).to(self.device)

        self.deepep_dispatcher = DeepEPDispatcher(
            group=process_group,
            router_topk=self.top_k + 1,
            permute_fusion=True,
            num_experts=self.num_experts,
            num_local_experts=self.num_experts // self.mp_size,
            hidden_size=self.hidden_size,
            params_dtype=torch.half,
            deepep_mode=DeepEPMode.normal,
            async_finish=True,
            return_recv_hook=True,
        )
        
        hidden_states = torch.randn(
            bsz * seq_len, self.hidden_size, device=self.device, dtype=torch.bfloat16
        )
        num_token_non_padded = torch.tensor([bsz * seq_len], device=self.device, dtype=torch.int32)
            
        def run_bench() -> torch.Tensor:
            router_logits = self.gate(hidden_states)
            topk_weights, topk_idx = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=True,
                renormalize=self.renormalize,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                correction_bias=self.correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                num_token_non_padded=num_token_non_padded,
            )
            # Dispatch tokens to experts.
            (hidden_states_, topk_idx, topk_weights,
             reorder_topk_ids, num_recv_tokens_per_expert,
             seg_indptr, masked_m, expected_m) = self.deepep_dispatcher.dispatch(
                hidden_states,
                topk_idx,
                topk_weights,
                forward_mode=ForwardMode.EXTEND,
            )
            final_hidden_states = experts(
                hidden_states=hidden_states_,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                reorder_topk_ids=reorder_topk_ids,
                seg_indptr=seg_indptr,
                masked_m=masked_m,
                expected_m=expected_m,
                num_recv_tokens_per_expert=num_recv_tokens_per_expert,
                forward_mode=ForwardMode.EXTEND,
            )
            # Combine expert outputs.
            final_hidden_states = self.deepep_dispatcher.combine(
                final_hidden_states,
                topk_idx,
                topk_weights,
                forward_mode=ForwardMode.EXTEND,
            )
            return final_hidden_states

        ms = bench(run_bench)
        return ms, 0, 0

    def bench_low_lat_deepep_moe(self, bsz: int, seq_len: int,
                                 process_group: torch.distributed.ProcessGroup) -> Tuple[float, int, int]:
        """
        Benchmark the low latency deepep MoE implementation.
        """
        if bsz * seq_len > 128:
            return 0.0, 0, 0
        from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
        from sglang.srt.utils import DeepEPMode
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.layers.moe.topk import select_experts
        from sglang.srt.layers.moe.ep_moe.token_dispatcher import DeepEPDispatcher

        experts = DeepEPMoE(
            layer_id=0,
            num_experts=self.num_experts,
            top_k=self.top_k + 1,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            renormalize=self.renormalize,
            quant_config=self.quant_config,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            tp_size=self.mp_size,
            deepep_mode=DeepEPMode.low_latency
        ).to(self.device)

        self.deepep_dispatcher = DeepEPDispatcher(
            group=process_group,
            router_topk=self.top_k + 1,
            permute_fusion=True,
            num_experts=self.num_experts,
            num_local_experts=self.num_experts // self.mp_size,
            hidden_size=self.hidden_size,
            params_dtype=torch.half,
            deepep_mode=DeepEPMode.low_latency,
            async_finish=True,
            return_recv_hook=True,
        )
        
        hidden_states = torch.randn(
            bsz * seq_len, self.hidden_size, device=self.device, dtype=torch.bfloat16
        )
        num_token_non_padded = torch.tensor([bsz * seq_len], device=self.device, dtype=torch.int32)
            
        def run_bench() -> torch.Tensor:
            router_logits = self.gate(hidden_states)
            topk_weights, topk_idx = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=True,
                renormalize=self.renormalize,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                correction_bias=self.correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                num_token_non_padded=num_token_non_padded,
            )
            (hidden_states_, topk_idx, topk_weights,
             reorder_topk_ids, num_recv_tokens_per_expert,
             seg_indptr, masked_m, expected_m) = self.deepep_dispatcher.dispatch(
                hidden_states,
                topk_idx,
                topk_weights,
                forward_mode=ForwardMode.EXTEND,
            )
            final_hidden_states = experts(
                hidden_states=hidden_states_,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                reorder_topk_ids=reorder_topk_ids,
                seg_indptr=seg_indptr,
                masked_m=masked_m,
                expected_m=expected_m,
                num_recv_tokens_per_expert=num_recv_tokens_per_expert,
                forward_mode=ForwardMode.EXTEND,
            )
            final_hidden_states = self.deepep_dispatcher.combine(
                final_hidden_states,
                topk_idx,
                topk_weights,
                forward_mode=ForwardMode.EXTEND,
            )
            return final_hidden_states

        ms = bench(run_bench)
        return ms, 0, 0


def run(local_rank: int, num_local_ranks: int, benchmark: str,
        bszs: List[int], seq_lens: List[int], config: Dict[str, Any]) -> None:
    """
    Entry point for distributed execution.

    Args:
        local_rank: Local GPU index.
        num_local_ranks: Total number of GPUs per node.
        benchmark: Benchmark function to run.
        bszs: List of batch sizes.
        seq_lens: List of sequence lengths.
        config: Configuration parameters loaded from JSON.
    """
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    moe_bench = MoEBenchmark(
        local_rank=local_rank,
        mp_size=num_ranks,
        **config
    )
    func = moe_bench.benchmark_funcs[benchmark]
    for seq_len in seq_lens:
        for bsz in bszs:
            ms, _, _ = func(bsz, seq_len, group)
            if rank == 0:
                print(f"Benchmark {benchmark} Rank {rank}, Batch size: {bsz}, "
                    f"Sequence length: {seq_len}, Time: {ms:.4f} sec")


def main() -> None:
    """Parse command line arguments and spawn multiprocessing workers for benchmarking."""
    parser = argparse.ArgumentParser(description="MoE Benchmark")
    parser.add_argument("--func", type=str,
                        choices=["fused_moe", "ep_moe", "normal_deepep_moe", "low_lat_deepep_moe"],
                        default="fused_moe",
                        help="Benchmark function to run")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1],
                        help="Batch sizes to benchmark")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[32, 512],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--mp-sizes", type=int, nargs="+", default=[8],
                        help="Tensor/Expert parallel sizes")
    args = parser.parse_args()

    func = args.func
    bszs = args.batch_sizes
    seq_lens = args.seq_lengths
    num_gpus = 8

    json_config_path = layer_configs.get("moe")
    with open(json_config_path, "r") as f:
        config: Dict[str, Any] = json.load(f)

    torch.multiprocessing.spawn(run, args=(num_gpus, func, bszs, seq_lens, config), nprocs=num_gpus)


if __name__ == "__main__":
    main()