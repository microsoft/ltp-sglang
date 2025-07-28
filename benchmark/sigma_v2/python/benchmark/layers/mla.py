"""
Benchmark different implementations of attention for LLMs.

This script compares performance of various attention implementations:
- Decode Attention:
    - FlashInfer MLA
    - Flash Attention 3
    - Flash MLA
- Prefill Attention:
    - FlashInfer
    - Flash Attention 3
"""
import os
import torch
import pandas as pd
from einops import rearrange
from triton.testing import do_bench
import flashinfer
from sgl_kernel.flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func
from flash_mla import flash_mla_with_kvcache, get_mla_metadata
from layer import LayerBenchmark


class MLAAttentionBenchmark(LayerBenchmark):
    """Benchmark different attention implementations for LLMs."""
    
    def __init__(
        self,
        page_size: int = 64,
        decode_head_dim_ckv: int = 512,
        decode_head_dim_kpe: int = 64,
        prefill_head_dim: int = 192,
        prefill_v_head_dim: int = 128,
        num_heads: int = 128,
        mp_size: int = 8,
    ):
        """
        Initialize the benchmark with configuration parameters.
        
        Args:
                page_size: Size of KV cache pages
                decode_head_dim_ckv: Content key-value head dimension for decode
                decode_head_dim_kpe: Positional embedding head dimension for decode
                prefill_head_dim: Head dimension for prefill attention
                prefill_v_head_dim: Value head dimension for prefill attention
                num_heads: Total number of attention heads
                mp_size: Model parallelism size
        """
        super().__init__("MLA")
        self.page_size = page_size
        
        # Decode attention parameters
        self.decode_head_dim = decode_head_dim_ckv + decode_head_dim_kpe
        self.decode_head_dim_ckv = decode_head_dim_ckv
        self.decode_head_dim_kpe = decode_head_dim_kpe
        
        # Prefill attention parameters
        self.prefill_head_dim = prefill_head_dim
        self.prefill_v_head_dim = prefill_v_head_dim
        
        # Parallelism settings
        self.num_heads = num_heads
        self.mp_size = mp_size
        self.num_heads_per_device = self.num_heads // self.mp_size
        
        # Set random seed for reproducibility
        torch.manual_seed(1123)
        
        # Define benchmark functions for each phase
        self.benchmark_funcs = {
                "decode_flashinfer": self.bench_flashinfer_decode,
                "decode_fa3": self.bench_fa3_decode,
                "decode_flashmla": self.bench_flashmla_decode,
                "prefill_flashinfer": self.bench_flashinfer_prefill,    
                "prefill_fa3": self.bench_fa3_prefill,
        }
            
    def bench_flashinfer_decode(self, bsz: int, seq_len: int) -> tuple:
        """
        Benchmark FlashInfer decode attention.
        
        Args:
                bsz: Batch size
                seq_len: Sequence length
                
        Returns:
                Tuple of (time_ms, io_gb_per_s, tflops)
        """
        try:
            # Create query and key-value tensors
            q_nope = torch.randn(bsz, self.num_heads_per_device, self.decode_head_dim_ckv, 
                                                    dtype=torch.half, device=self.device)
            q_pe = torch.randn(bsz, self.num_heads_per_device, self.decode_head_dim_kpe, 
                                                dtype=torch.half, device=self.device)
            ckv = torch.randn(bsz * seq_len, 1, self.decode_head_dim_ckv, 
                                                dtype=torch.half, device=self.device)
            kpe = torch.randn(bsz * seq_len, 1, self.decode_head_dim_kpe, 
                                                dtype=torch.half, device=self.device)
            
            # Scale factor for softmax
            sm_scale = 1.0 / ((self.decode_head_dim_ckv + self.decode_head_dim_kpe) ** 0.5)
            
            # Setup FlashInfer workspace and wrapper
            workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=self.device)
            decode_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace_buffer, backend="fa3")
            
            # Setup index pointers for batch processing
            q_indptr = torch.arange(0, bsz + 1, device=self.device, dtype=torch.int)
            kv_lens = torch.full((bsz,), seq_len, dtype=torch.int32, device=self.device)
            kv_indptr = torch.arange(0, bsz + 1, device=self.device, dtype=torch.int) * seq_len
            kv_indices = torch.arange(0, bsz * seq_len, device=self.device, dtype=torch.int)
            
            # Plan the computation
            decode_wrapper.plan(
                    q_indptr,
                    kv_indptr,
                    kv_indices,
                    kv_lens,
                    self.num_heads_per_device,
                    self.decode_head_dim_ckv,
                    self.decode_head_dim_kpe,
                    self.page_size,
                    False,  # causal
                    sm_scale,
                    q_nope.dtype,
                    ckv.dtype,
            )
            
            # Initial run to ensure everything is loaded
            o = decode_wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)
            
            # Benchmark run
            ms = do_bench(lambda: decode_wrapper.run(q_nope, q_pe, ckv, kpe), 
                                        warmup=self.warmup_iters, rep=self.benchmark_iters)
            
            # Calculate metrics
            io_gb_per_s = self._calculate_io_gb_per_s([q_nope, q_pe, ckv, kpe, o], ms)
            tflops = self._calculate_decode_tflops(bsz, seq_len, ms)
                
            return ms, io_gb_per_s, tflops
                    
        except Exception as e:
            print(f"Error in flashinfer decode benchmark: {e}")
            return float('inf'), 0, 0

    def bench_fa3_decode(self, bsz: int, seq_len: int) -> tuple:
        """
        Benchmark Flash Attention 3 decode attention.
        
        Args:
                bsz: Batch size
                seq_len: Sequence length
                
        Returns:
                Tuple of (time_ms, io_gb_per_s, tflops)
        """
        try:
            # Setup sequence lengths and page table
            cache_seqlens = torch.tensor([seq_len] * bsz, device=self.device, dtype=torch.int)
            num_splits = 0
            
            # Create page table for KV cache
            page_table = rearrange(
                    torch.arange(bsz * seq_len // self.page_size, device=self.device, dtype=torch.int32), 
                    "(b s) -> b s", 
                    s=seq_len // self.page_size
            )
            
            # Create input tensors
            q_nope = torch.randn(bsz, 1, self.num_heads_per_device, self.decode_head_dim_ckv, 
                                                    dtype=torch.half, device=self.device)
            q_pe = torch.randn(bsz, 1, self.num_heads_per_device, self.decode_head_dim_kpe, 
                                                dtype=torch.half, device=self.device)
            ckv = torch.randn(bsz, seq_len, 1, self.decode_head_dim_ckv, 
                                            dtype=torch.half, device=self.device)
            kpe = torch.randn(bsz, seq_len, 1, self.decode_head_dim_kpe, 
                                            dtype=torch.half, device=self.device)
            
            # Rearrange KV cache for paged format
            k_cache, v_cache = [rearrange(x, "b (n p) h d -> (b n) p h d", p=self.page_size) for x in [kpe, ckv]]
            
            
            # Define benchmark function
            def run_fa3():
                    return flash_attn_with_kvcache(
                            q=q_pe, 
                            k_cache=k_cache,
                            v_cache=v_cache,
                            qv=q_nope,
                            cache_seqlens=cache_seqlens, 
                            num_splits=num_splits, 
                            page_table=page_table, 
                            causal=False,
                    )
            
            # Initial run to ensure everything is loaded
            o = run_fa3()
            
            # Benchmark run
            ms = do_bench(run_fa3, warmup=self.warmup_iters, rep=self.benchmark_iters)
            
            # Calculate metrics
            io_gb_per_s = self._calculate_io_gb_per_s([q_nope, q_pe, ckv, kpe, o], ms)
            tflops = self._calculate_decode_tflops(bsz, seq_len, ms)
            
            return ms, io_gb_per_s, tflops
            
        except Exception as e:
            print(f"Error in FA3 decode benchmark: {e}")
            return float('inf'), 0, 0

    def bench_flashmla_decode(self, bsz: int, seq_len: int) -> tuple:
        """
        Benchmark Flash MLA decode attention.
        
        Args:
                bsz: Batch size
                seq_len: Sequence length
                
        Returns:
                Tuple of (time_ms, io_gb_per_s, tflops)
        """
        try:
            # Create concatenated query tensor
            q_concat = torch.randn(bsz, 1, self.num_heads_per_device, self.decode_head_dim, 
                                                        dtype=torch.half, device=self.device)
            
            # Create page table for KV cache
            page_table = rearrange(
                    torch.arange(seq_len * bsz // self.page_size, device=self.device, dtype=torch.int32), 
                    "(b s) -> b s", 
                    s=seq_len // self.page_size
            )
            
            # Setup sequence lengths and metadata
            cache_seqlens = torch.tensor([seq_len] * bsz, device=self.device, dtype=torch.int)
            tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, 128, 1)
            
            # Create KV cache tensor
            kv_cache_concat = torch.randn(page_table.numel(), self.page_size, 1, self.decode_head_dim, 
                                                                    dtype=torch.half, device=self.device)
            
            # Define benchmark function
            def run_flashmla():
                    return flash_mla_with_kvcache(
                            q_concat,
                            kv_cache_concat, 
                            page_table,
                            cache_seqlens, 
                            self.decode_head_dim_ckv, 
                            tile_scheduler_metadata, 
                            num_splits,
                            causal=False
                    )
            
            # Initial run to ensure everything is loaded
            o = run_flashmla()
            
            # Use fewer iterations for FlashMLA due to potential memory pressure
            ms = do_bench(run_flashmla, warmup=1, rep=10)
            
            # Calculate metrics
            io_gb_per_s = self._calculate_io_gb_per_s([q_concat, kv_cache_concat, o[0], o[1]], ms)
            tflops = self._calculate_decode_tflops(bsz, seq_len, ms)
            
            return ms, io_gb_per_s, tflops
                
        except Exception as e:
            print(f"Error in FlashMLA decode benchmark: {e}")
            return float('inf'), 0, 0

    def bench_flashinfer_prefill(self, bsz: int, seq_len: int) -> tuple:
        """
        Benchmark FlashInfer prefill attention.
        
        Args:
                bsz: Batch size
                seq_len: Sequence length
                
        Returns:
                Tuple of (time_ms, io_gb_per_s, tflops)
        """
        try:
            # Create input tensors
            q = torch.randn(bsz * seq_len, self.num_heads_per_device, self.prefill_head_dim, 
                                        dtype=torch.half, device=self.device)
            k = torch.randn(bsz * seq_len, self.num_heads_per_device, self.prefill_head_dim, 
                                        dtype=torch.half, device=self.device)
            v = torch.randn(bsz * seq_len, self.num_heads_per_device, self.prefill_v_head_dim, 
                                        dtype=torch.half, device=self.device)
            
            # Setup FlashInfer workspace and wrapper
            workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=self.device)
            prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                    workspace_buffer, "NHD", backend="fa3"
            )
            
            # Setup index pointers for batch processing
            qo_indptr = torch.arange(0, bsz * seq_len + 1, seq_len).int()
            kv_indptr = torch.arange(0, bsz * seq_len + 1, seq_len).int()
            
            # Plan the computation
            prefill_wrapper.plan(
                    qo_indptr,
                    kv_indptr,
                    self.num_heads_per_device,
                    self.num_heads_per_device,
                    self.prefill_head_dim,
                    self.prefill_v_head_dim,
                    causal=False,
            )
            
            # Initial run to ensure everything is loaded
            o = prefill_wrapper.run(q, k, v)
            
            # Benchmark run
            ms = do_bench(lambda: prefill_wrapper.run(q, k, v), 
                                        warmup=self.warmup_iters, rep=self.benchmark_iters)
            
            # Calculate metrics
            io_gb_per_s = self._calculate_io_gb_per_s([q, k, v, o], ms)
            tflops = self._calculate_prefill_tflops(bsz, seq_len, ms)
            
            return ms, io_gb_per_s, tflops
                
        except Exception as e:
            print(f"Error in flashinfer prefill benchmark: {e}")
            return float('inf'), 0, 0

    def bench_fa3_prefill(self, bsz: int, seq_len: int) -> tuple:
        """
        Benchmark Flash Attention 3 prefill attention.
        
        Args:
                bsz: Batch size
                seq_len: Sequence length
                
        Returns:
                Tuple of (time_ms, io_gb_per_s, tflops)
        """
        try:
            # Create input tensors
            q = torch.randn(bsz * seq_len, self.num_heads_per_device, self.prefill_head_dim,
                                        dtype=torch.half, device=self.device)
            k = torch.randn(bsz * seq_len, self.num_heads_per_device, self.prefill_head_dim,
                                        dtype=torch.half, device=self.device)
            v = torch.randn(bsz * seq_len, self.num_heads_per_device, self.prefill_v_head_dim,
                                        dtype=torch.half, device=self.device)
            
            # Setup sequence lengths for batch processing
            cu_seqlens_q = torch.arange(bsz + 1, device=self.device, dtype=torch.int32) * seq_len
            cu_seqlens_k = torch.arange(bsz + 1, device=self.device, dtype=torch.int32) * seq_len
            
            # Define benchmark function
            def run_fa3():
                    return flash_attn_varlen_func(
                            q=q, 
                            k=k,
                            v=v,
                            cu_seqlens_q=cu_seqlens_q,
                            cu_seqlens_k=cu_seqlens_k,
                            max_seqlen_q=seq_len,
                            max_seqlen_k=seq_len,
                            causal=False,
                    )
            
            # Initial run to ensure everything is loaded
            o = run_fa3()
            
            # Benchmark run
            ms = do_bench(run_fa3, warmup=self.warmup_iters, rep=self.benchmark_iters)
            
            # Calculate metrics
            io_gb_per_s = self._calculate_io_gb_per_s([q, k, v, o[0], o[1]], ms)
            tflops = self._calculate_prefill_tflops(bsz, seq_len, ms)
            
            return ms, io_gb_per_s, tflops
                
        except Exception as e:
            print(f"Error in FA3 prefill benchmark: {e}")
            return float('inf'), 0, 0

    def _calculate_prefill_tflops(self, bsz: int, seq_len: int, ms: float) -> float:
        """
        Helper method to calculate TFLOPS for MoE operations.
        
        Args:
            bsz: Batch size
            seq_len: Sequence length
            ms: Latency in milliseconds
            
        Returns:
            Throughput in TFLOPS
        """
        # print("prefill", 2 * bsz * seq_len * seq_len * self.num_heads_per_device * (self.prefill_head_dim + self.prefill_v_head_dim))
        return 2 * bsz * seq_len * seq_len * self.num_heads_per_device * (self.prefill_head_dim + self.prefill_v_head_dim) * 1e-12 / (ms * 1e-3)
    
    def _calculate_decode_tflops(self, bsz: int, seq_len: int, ms: float) -> float:
        """
        Helper method to calculate TFLOPS for decode operations.
        
        Args:
            bsz: Batch size
            seq_len: Sequence length
            ms: Latency in milliseconds
            
        Returns:
            Throughput in TFLOPS
        """
        # print("decode", 2 * bsz * seq_len * self.num_heads_per_device * (2 * self.decode_head_dim_ckv + self.decode_head_dim_kpe))
        return 2 * bsz * seq_len * self.num_heads_per_device * (2 * self.decode_head_dim_ckv + self.decode_head_dim_kpe) * 1e-12 / (ms * 1e-3)
    
    def _calculate_io_gb_per_s(self, tensors, ms):
        """
        Helper method to calculate I/O bandwidth in GB/s.
        
        Args:
            tensors: List of tensors involved in the operation
            ms: Latency in milliseconds
            
        Returns:
            Bandwidth in GB/s
        """
        io_bytes = sum(x.numel() * x.element_size() for x in tensors)
        return io_bytes * 1e-9 / (ms * 1e-3)
    
        
def prepare_benchmark_data(results):
        """
        Prepare benchmark data for display and analysis.
        
        Args:
                results: Dictionary of benchmark results
                
        Returns:
                Dictionary of DataFrames organized by phase
        """
        data_by_phase = {}
        
        for phase, phase_results in results.items():        
                data = []
                
                for config, methods in phase_results.items():
                        # Parse configuration string
                        parts = config.split('_')
                        batch_size = parts[1]
                        seq_len = parts[3]
                        tp = parts[5]
                        
                        # Find the fastest method for this configuration
                        fastest = min(methods.items(), key=lambda x: x[1]["time_ms"])
                        fastest_ms = fastest[1]["time_ms"]
                        
                        # Add data for each method
                        for method, metrics in methods.items():
                                speedup = fastest_ms / metrics["time_ms"]
                                speedup_str = "fastest" if method == fastest[0] else f"{speedup:.2f}x slower"
                                
                                data.append({
                                        "Bsz": int(batch_size),
                                        "Seq": int(seq_len),
                                        "TP": int(tp),
                                        "Method": method,
                                        "Time (ms)": metrics["time_ms"],
                                        "GB/s": metrics["bandwidth_gb_s"],
                                        "TFLOPs": metrics["throughput_tflops"],
                                        "Speedup": speedup_str
                                })
                
                data_by_phase[phase] = data
        
        return data_by_phase


def load_existing_results(phase):
        """
        Check if results already exist and load them.
        
        Args:
                phase: Benchmark phase ("decode", "prefill", or "both")
                
        Returns:
                DataFrame of results or None if no existing results
        """
        csv_path = f"{phase}_attention_benchmark_raw_results.csv"
        if os.path.exists(csv_path):
                print(f"Loading existing results from {csv_path}")
                return pd.read_csv(csv_path)
        return None