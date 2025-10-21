import os
import socket
import time
from typing import List, Tuple

import ray
import torch
import torch.distributed as dist

from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    graph_capture,
    initialize_model_parallel,
)


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
    test_sizes: List[int],
    dtypes: List[torch.dtype],
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
    
    for sz in test_sizes:
        for dtype in dtypes:
            dtype_name = str(dtype).split('.')[-1]
            
            # Warmup
            for _ in range(warmup_iters):
                data = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                dist.all_reduce(data, group=group)
                torch.cuda.synchronize()
            
            # Benchmark NCCL allreduce
            nccl_times = []
            for _ in range(benchmark_iters):
                data = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                torch.cuda.synchronize()
                start = time.perf_counter()
                dist.all_reduce(data, group=group)
                torch.cuda.synchronize()
                end = time.perf_counter()
                nccl_times.append((end - start) * 1000)  # Convert to ms
            
            # Warmup custom allreduce
            for _ in range(warmup_iters):
                data = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                _ = tensor_model_parallel_all_reduce(data)
                torch.cuda.synchronize()
            
            # Benchmark custom allreduce
            custom_times = []
            for _ in range(benchmark_iters):
                data = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = tensor_model_parallel_all_reduce(data)
                torch.cuda.synchronize()
                end = time.perf_counter()
                custom_times.append((end - start) * 1000)  # Convert to ms
            
            # Calculate statistics
            nccl_avg = sum(nccl_times) / len(nccl_times)
            nccl_min = min(nccl_times)
            nccl_max = max(nccl_times)
            
            custom_avg = sum(custom_times) / len(custom_times)
            custom_min = min(custom_times)
            custom_max = max(custom_times)
            
            results.append({
                'rank': rank,
                'size': sz,
                'dtype': dtype_name,
                'size_bytes': sz * torch.tensor([], dtype=dtype).element_size(),
                'nccl_avg_ms': nccl_avg,
                'nccl_min_ms': nccl_min,
                'nccl_max_ms': nccl_max,
                'custom_avg_ms': custom_avg,
                'custom_min_ms': custom_min,
                'custom_max_ms': custom_max,
                'speedup': nccl_avg / custom_avg if custom_avg > 0 else 0,
            })
    
    return results


@ray.remote(num_gpus=1, max_calls=1)
def benchmark_graph_allreduce(
    world_size: int,
    rank: int,
    distributed_init_port: int,
    test_sizes: List[int],
    dtypes: List[torch.dtype],
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
    
    for sz in test_sizes:
        for dtype in dtypes:
            dtype_name = str(dtype).split('.')[-1]
            
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
                graph_times.append((end - start) * 1000)
            
            graph_avg = sum(graph_times) / len(graph_times)
            graph_min = min(graph_times)
            graph_max = max(graph_times)
            
            results.append({
                'rank': rank,
                'size': sz,
                'dtype': dtype_name,
                'size_bytes': sz * torch.tensor([], dtype=dtype).element_size(),
                'graph_avg_ms': graph_avg,
                'graph_min_ms': graph_min,
                'graph_max_ms': graph_max,
            })
    
    return results


def print_results(all_results: List[dict], mode: str = "eager"):
    """Pretty print benchmark results."""
    if not all_results:
        return
    
    print(f"\n{'='*100}")
    print(f"AllReduce Latency Benchmark Results ({mode.upper()} mode)")
    print(f"{'='*100}")
    
    # Group by size and dtype
    from collections import defaultdict
    grouped = defaultdict(list)
    for result in all_results:
        key = (result['size'], result['dtype'])
        grouped[key].append(result)
    
    if mode == "eager":
        print(f"{'Size':<12} {'Dtype':<10} {'Size(Bytes)':<15} {'NCCL(ms)':<20} {'Custom(ms)':<20} {'Speedup':<10}")
        print(f"{'-'*100}")
        
        for (size, dtype), results in sorted(grouped.items()):
            # Average across ranks
            nccl_avg = sum(r['nccl_avg_ms'] for r in results) / len(results)
            custom_avg = sum(r['custom_avg_ms'] for r in results) / len(results)
            speedup = sum(r['speedup'] for r in results) / len(results)
            size_bytes = results[0]['size_bytes']
            
            print(f"{size:<12} {dtype:<10} {size_bytes:<15} {nccl_avg:<20.4f} {custom_avg:<20.4f} {speedup:<10.2f}x")
    
    elif mode == "graph":
        print(f"{'Size':<12} {'Dtype':<10} {'Size(Bytes)':<15} {'Graph Avg(ms)':<20} {'Graph Min(ms)':<20} {'Graph Max(ms)':<20}")
        print(f"{'-'*100}")
        
        for (size, dtype), results in sorted(grouped.items()):
            graph_avg = sum(r['graph_avg_ms'] for r in results) / len(results)
            graph_min = min(r['graph_min_ms'] for r in results)
            graph_max = max(r['graph_max_ms'] for r in results)
            size_bytes = results[0]['size_bytes']
            
            print(f"{size:<12} {dtype:<10} {size_bytes:<15} {graph_avg:<20.4f} {graph_min:<20.4f} {graph_max:<20.4f}")


def main():
    # Configuration
    WORLD_SIZES = [8]
    TEST_SIZES = [1024 * i for i in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]]
    DTYPES = [torch.bfloat16]
    WARMUP_ITERS = 10
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
        
        # Eager mode benchmark
        # print(f"\nRunning eager mode benchmark...")
        # refs = []
        # for rank in range(world_size):
        #     refs.append(
        #         benchmark_allreduce.remote(
        #             world_size, rank, distributed_init_port,
        #             TEST_SIZES, DTYPES, WARMUP_ITERS, BENCHMARK_ITERS
        #         )
        #     )
        # eager_results = []
        # for result in ray.get(refs):
        #     eager_results.extend(result)
        # print_results(eager_results, mode="eager")
        
        # Graph mode benchmark
        # distributed_init_port = get_open_port()
        print(f"\nRunning graph mode benchmark...")
        refs = []
        for rank in range(world_size):
            refs.append(
                benchmark_graph_allreduce.remote(
                    world_size, rank, distributed_init_port,
                    TEST_SIZES, DTYPES, WARMUP_ITERS, BENCHMARK_ITERS
                )
            )
        graph_results = []
        for result in ray.get(refs):
            graph_results.extend(result)
        print_results(graph_results, mode="graph")
    
    ray.shutdown()
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
