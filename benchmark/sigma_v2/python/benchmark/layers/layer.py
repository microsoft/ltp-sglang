import time
from constants import warmup_iters, benchmark_iters, device
from tqdm import tqdm

class LayerBenchmark:
    def __init__(self, name: str):
        self.name = name
        self.device = device
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters

    def run_benchmarks(self, batch_sizes, seq_lens):
        """
        Run benchmarks with the given input sizes.
        
        Args:
            batch_sizes: List of batch sizes to benchmark
            seq_lens: List of sequence lengths to benchmark
            
        Returns:
            Dictionary of benchmark results
        """
        results = {}
                
        total_iterations = len(seq_lens) * len(batch_sizes)
        progress_bar = tqdm(total=total_iterations, desc=f"Benchmark {self.name}", unit="bsz seq")
                
        for seq_len in seq_lens:
            for bsz in batch_sizes:
                config_key = (bsz, seq_len)
                results[config_key] = {}
                    
                # Run all benchmarks for this type
                for name, func in self.benchmark_funcs.items():
                    ms, gb_s, tflops = func(bsz, seq_len)
                    results[config_key][name] = {
                        "time_ms": ms,
                        "bandwidth_gb_s": gb_s,
                        "throughput_tflops": tflops
                    }
                    time.sleep(1)  # Avoid power throttling
                    
                progress_bar.update(1)
                
        progress_bar.close()
        return results