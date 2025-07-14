import torch
from triton.testing import do_bench
from layer import LayerBenchmark

class LinearBenchmark(LayerBenchmark):
    """Base class for linear layer benchmarks supporting FP16 and FP8 precision."""
    
    def __init__(
        self,
        name: str = "Linear",
        hidden_size: int = 7168,
        intermediate_size: int = 2048,
        mp_size: int = 1,
    ):
        """
        Initialize the linear benchmark.
        
        Args:
            hidden_size: Input dimension size
            intermediate_size: Output dimension size
            mp_size: Model parallelism size
        """
        super().__init__(name)
        # Apply model parallelism settings
        import mp_patch
        mp_patch.apply_hardcode_parallel(mp_size)
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.mp_size = mp_size
        
        # Setup FP8 quantization configuration
        from sglang.srt.layers.quantization.fp8 import Fp8Config
        self.quant_config = Fp8Config.from_config({
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128]
        })
        
        # Register benchmark functions
        self.benchmark_funcs = {
            "fp16_linear": self.bench_fp16_linear,
            "fp8_linear": self.bench_fp8_linear,
            "fp16_col_linear": self.bench_fp16_col_linear,
            "fp8_col_linear": self.bench_fp8_linear,
            "fp16_row_linear": self.bench_fp16_row_linear,
            "fp8_row_linear": self.bench_fp8_linear,
        }
    
    def bench_fp16_linear(self, bsz: int, seq_len: int, 
                          col_parallel: bool = False,
                          rol_parallel: bool = False) -> tuple:
        """
        Benchmark FP16 linear layer.
        
        Args:
            bsz: Batch size
            seq_len: Sequence length
            
        Returns:
            Tuple of (latency_ms, memory_usage, tflops)
        """
        try:
            from sglang.srt.layers.linear import ReplicatedLinear
            
            # Create linear layer without quantization
            hidden_size = self.hidden_size // self.mp_size if col_parallel else self.hidden_size
            intermediate_size = self.intermediate_size // self.mp_size if rol_parallel else self.intermediate_size
            linear = ReplicatedLinear(
                input_size=hidden_size,
                output_size=intermediate_size,
                params_dtype=torch.float16,
                bias=False,
                quant_config=None
            ).to(self.device)
            linear.eval()
            
            # Create input tensor
            hidden_states = torch.randn(
                bsz * seq_len, hidden_size, 
                device=self.device, 
                dtype=torch.float16
            )
            
            def run():
                with torch.no_grad():
                    output = linear(hidden_states)
                return output
            
            # Benchmark performance
            ms = do_bench(run, warmup=self.warmup_iters, rep=self.benchmark_iters)
            
            # Calculate TFLOPS: 2 operations per element (multiply-add) * input size * output size
            tflops = (2 * bsz * seq_len * hidden_size * intermediate_size) * 1e-12 / (ms * 1e-3)
            return ms, 0, tflops
        
        except Exception as e:
            print(f"Error in FP16 {self.name} benchmark: {e}")
            return float('inf'), 0, 0
    
    def bench_fp16_col_linear(self, bsz: int, seq_len: int) -> tuple:
        return self.bench_fp16_linear(bsz, seq_len, col_parallel=True)
    
    def bench_fp16_row_linear(self, bsz: int, seq_len: int) -> tuple:
        return self.bench_fp16_linear(bsz, seq_len, rol_parallel=True)
    
    def bench_fp8_linear(self, bsz: int, seq_len: int,
                          col_parallel: bool = False,
                          rol_parallel: bool = False) -> tuple:
        """
        Benchmark FP8 quantized linear layer.
        
        Args:
            bsz: Batch size
            seq_len: Sequence length
            
        Returns:
            Tuple of (latency_ms, memory_usage, tflops)
        """
        try:
            from sglang.srt.layers.linear import ReplicatedLinear
            hidden_size = self.hidden_size // self.mp_size if col_parallel else self.hidden_size
            intermediate_size = self.intermediate_size // self.mp_size if rol_parallel else self.intermediate_size
            # Create linear layer with FP8 quantization
            linear = ReplicatedLinear(
                input_size=hidden_size,
                output_size=intermediate_size,
                bias=False,
                quant_config=self.quant_config
            ).to(self.device)
            linear.eval()
            
            # Create input tensor (still in FP16)
            hidden_states = torch.randn(
                bsz * seq_len, hidden_size, 
                device=self.device, 
                dtype=torch.bfloat16)
            
            def run():
                with torch.no_grad():
                    output = linear(hidden_states)
                return output
            
            # Benchmark performance
            ms = do_bench(run, warmup=self.warmup_iters, rep=self.benchmark_iters)
            
            # Calculate TFLOPS
            tflops = (2 * bsz * seq_len * hidden_size * intermediate_size) * 1e-12 / (ms * 1e-3)
            return ms, 0, tflops
        
        except Exception as e:
            print(f"Error in FP8 {self.name} benchmark: {e}")
            return float('inf'), 0, 0

    def bench_fp8_col_linear(self, bsz: int, seq_len: int) -> tuple:
        return self.bench_fp8_linear(bsz, seq_len, col_parallel=True)
    
    def bench_fp8_row_linear(self, bsz: int, seq_len: int) -> tuple:
        return self.bench_fp8_linear(bsz, seq_len, rol_parallel=True)
