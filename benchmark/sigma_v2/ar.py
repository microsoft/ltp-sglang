import os
import argparse
import socket
import time
from typing import List

import ray
import torch
import torch.distributed as dist

from sglang.srt.distributed import init_distributed_environment
import pandas as pd
from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
)

HIDDEN_SIZE = 5120

def get_open_port() -> int:
    """Get an available port for distributed communication."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


@ray.remote(num_gpus=1, max_calls=1)
def benchmark_allreduce(
    world_size: int,
    rank: int,
    distributed_init_port: int,
    phases: List[str],
    bszs: List[int],
    seq_lens: int,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
):
    """Benchmark both NCCL and custom allreduce operations."""
    
    # Setup distributed environment
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
        local_rank=rank,
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    group = get_tensor_model_parallel_group().device_group

    # Warmup NCCL
    warmup_data = torch.zeros(1, device=device)
    torch.distributed.all_reduce(warmup_data, group=group)
    torch.cuda.synchronize()
    
    results = []
    
    for phase in phases:
        for bsz in bszs:
            sz = bsz * HIDDEN_SIZE * seq_lens if phase == 'prefill' else bsz * HIDDEN_SIZE
            dtype = torch.bfloat16
            for _ in range(warmup_iters):
                data = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                dist.all_reduce(data, group=group)
                torch.cuda.synchronize()
            
            # Benchmark every run of NCCL allreduce
            nccl_times = []
            data = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
            for _ in range(benchmark_iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                dist.all_reduce(data, group=group)
                torch.cuda.synchronize()
                end = time.perf_counter()
                nccl_times.append((end - start) * 1000 * 1000)  # Convert to us
            
            # Benchmark average over multiple runs of NCCL allreduce
            data = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(benchmark_iters):
                dist.all_reduce(data, group=group)
            torch.cuda.synchronize()
            end = time.perf_counter()
            avg_nccl_time = (end - start) / benchmark_iters * 1000 * 1000  # Convert to us
            
            # Warmup custom allreduce
            data = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
            for _ in range(warmup_iters):
                _ = tensor_model_parallel_all_reduce(data)
            torch.cuda.synchronize()
            
            # Benchmark custom allreduce
            custom_times = []
            avg_custom_time = 0.0
            data = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
            for _ in range(benchmark_iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = tensor_model_parallel_all_reduce(data)
                torch.cuda.synchronize()
                end = time.perf_counter()
                custom_times.append((end - start) * 1000 * 1000)  # Convert to us
            
            data = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(benchmark_iters):
                _ = tensor_model_parallel_all_reduce(data)
            torch.cuda.synchronize()
            end = time.perf_counter()
            avg_custom_time = (end - start) / benchmark_iters * 1000 * 1000  # Convert to us
            
            
            # Calculate statistics
            nccl_avg = sum(nccl_times) / len(nccl_times)
            nccl_min = min(nccl_times)
            nccl_max = max(nccl_times)
            
            custom_avg = sum(custom_times) / len(custom_times)
            custom_min = min(custom_times)
            custom_max = max(custom_times)
            
            results.append({
                'rank': rank,
                'phase': phase,
                'bsz': bsz,
                'size_bytes': sz * torch.tensor([], dtype=dtype).element_size(),
                'nccl_avg_us': nccl_avg,
                'nccl_min_us': nccl_min,
                'nccl_max_us': nccl_max,
                'nccl_avg_run_us': avg_nccl_time,
                'custom_avg_us': custom_avg,
                'custom_min_us': custom_min,
                'custom_max_us': custom_max,
                'custom_avg_run_us': avg_custom_time,
            })
    
    return results


@ray.remote(num_gpus=1, max_calls=1)
def benchmark_graph_allreduce(
    world_size: int,
    rank: int,
    distributed_init_port: int,
    phases: List[str],
    bszs: List[int],
    seq_lens: int,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
):
    """Benchmark allreduce operations with CUDA graph capture."""
    
    # Setup distributed environment
    del os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
        local_rank=rank,
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    group = get_tensor_model_parallel_group().device_group

    # Warmup
    warmup_data = torch.zeros(1, device=device)
    torch.distributed.all_reduce(warmup_data, group=group)
    torch.cuda.synchronize()
    
    results = []
    
    for phase in phases:
        for bsz in bszs:
            sz = bsz * HIDDEN_SIZE * seq_lens if phase == 'prefill' else bsz * HIDDEN_SIZE
            dtype = torch.bfloat16

            # Capture graph ONCE outside the benchmark loop
            with graph_capture() as graph_capture_context:
                inp = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                torch.cuda.synchronize()
                
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                    out = tensor_model_parallel_all_reduce(inp)
            
            # Warmup replays
            for _ in range(warmup_iters):
                graph.replay()
                torch.cuda.synchronize()
            
            # Benchmark graph replay only
            graph_times = []
            for _ in range(benchmark_iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                end = time.perf_counter()
                graph_times.append((end - start) * 1000 * 1000)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(benchmark_iters):
                graph.replay()
            torch.cuda.synchronize()
            end = time.perf_counter()
            graph_avg_time = (end - start) * 1000 * 1000 / benchmark_iters  # Convert to us
            
            graph_avg = sum(graph_times) / len(graph_times)
            graph_min = min(graph_times)
            graph_max = max(graph_times)
            
            results.append({
                'rank': rank,
                'phase': phase,
                'bsz': bsz,
                'size_bytes': sz * torch.tensor([], dtype=dtype).element_size(),
                'graph_avg_us': graph_avg,
                'graph_min_us': graph_min,
                'graph_max_us': graph_max,
                'graph_avg_run_us': graph_avg_time,
            })
    
    return results


def main(mode: str):
    # Configuration
    WORLD_SIZES = [8]
    PHASES = ['decode']
    BSZS = [1, 2, 4, 8, 16, 32]
    SEQ_LENS = 500
    WARMUP_ITERS = 100
    BENCHMARK_ITERS = 100
    
    ray.init(log_to_driver=True)
    
    for world_size in WORLD_SIZES:
        if world_size > torch.cuda.device_count():
            print(f"Skipping world_size={world_size} (not enough GPUs)")
            continue
        
        print(f"\n{'#'*100}")
        print(f"# Benchmarking with world_size={world_size}")
        print(f"{'#'*100}")
        
        distributed_init_port = get_open_port()
        results = []
        if mode == 'eager':
            # Eager mode benchmark
            print(f"\nRunning eager mode benchmark...")
            refs = []
            for rank in range(world_size):
                refs.append(
                    benchmark_allreduce.remote(
                        world_size, rank, distributed_init_port,
                        PHASES, BSZS, SEQ_LENS, WARMUP_ITERS, BENCHMARK_ITERS
                    )
                )
            for result in ray.get(refs):
                results.extend(result)
        
        if  mode == 'graph':
            # Graph mode benchmark
            print(f"\nRunning graph mode benchmark...")
            refs = []
            for rank in range(world_size):
                refs.append(
                    benchmark_graph_allreduce.remote(
                        world_size, rank, distributed_init_port,
                        PHASES, BSZS, SEQ_LENS, WARMUP_ITERS, BENCHMARK_ITERS
                    )
                )
            
            for result in ray.get(refs):
                results.extend(result)
            
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
    ray.shutdown()
    print("\nBenchmark completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AllReduce Benchmarking with SGLang DeepDive LTP")
    parser.add_argument('--mode', type=str, choices=['eager', 'graph'], default='eager', help='Benchmark mode to run')
    args = parser.parse_args()
    main(args.mode)
