"""
Benchmark different implementations of Mixture of Experts (MoE) for LLMs.
"""
import torch
from triton.testing import do_bench
from layer import LayerBenchmark

class ExpBenchmark(LayerBenchmark):
    """Benchmark different MoE implementations for LLMs."""
    
    def __init__(
        self,
        num_experts: int = 256,
        top_k: int = 8,
        hidden_size: int = 7168,
        intermediate_size: int = 2048,
        renormalize: bool = True,
        use_grouped_topk: bool = True,
        num_expert_group: int = 8,
        topk_group: int = 4,
        mp_size: int = 8,
        use_fp16: bool = False,
    ):
        """
        Initialize the benchmark with configuration parameters.
        
        Args:
            num_experts: Number of experts in MoE
            top_k: Number of experts to route each token to
            hidden_size: Hidden dimension size
            intermediate_size: Intermediate dimension size
            renormalize: Whether to renormalize expert outputs
            use_grouped_topk: Whether to use grouped topk routing
            num_expert_group: Number of expert groups
            topk_group: Number of experts per group
            mp_size: Tensor/Expert parallelism factor
            use_fp16: Whether to use FP16 instead of FP8
        """
        super().__init__("Exp")
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.use_fp8 = not use_fp16
        self.mp_size = mp_size
        
        # Setup model parallelism
        import mp_patch
        mp_patch.apply_hardcode_parallel(mp_size) 
        
        # Set random seed for reproducibility
        torch.manual_seed(27)
        
        # Initialize FP8 quantization if needed
        self.quant_config = None
        if self.use_fp8:
            from sglang.srt.layers.quantization.fp8 import Fp8Config
            self.quant_config = Fp8Config.from_config({
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128]
            })
            
        # Define benchmark functions
        self.benchmark_funcs = {
            "fused_moe": self.bench_fused_moe,
            "ep_moe": self.bench_ep_moe,
            "normal_deepep_moe": self.bench_normal_deepep_moe,
            "low_lat_deepep_moe": self.bench_low_lat_deepep_moe,
        }
    
    def bench_fused_moe(self, bsz: int, seq_len: int) -> tuple:
        """
        Benchmark FusedMoE implementation - a single fused kernel for MoE operations.
        
        Args:
            bsz: Batch size
            seq_len: Sequence length
            
        Returns:
            Tuple of (latency_ms, throughput_tokens_per_s, throughput_tflops)
        """
        try:
            # Import and initialize the MoE model
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
            
            # Prepare input tensors
            hidden_states = torch.randn(bsz * seq_len, self.hidden_size, dtype=torch.half, device=self.device)
            router_logits = torch.randn(bsz * seq_len, self.num_experts, dtype=torch.half, device=self.device)
            
            # Define the forward pass function
            def run_moe():
                return experts(hidden_states=hidden_states, router_logits=router_logits)
            
            # Measure performance
            ms = do_bench(run_moe, warmup=self.warmup_iters, rep=self.benchmark_iters)
            
            # Calculate TFLOPS (trillion floating-point operations per second)
            tflops = self._calculate_tflops(bsz, seq_len, ms)
            
            return ms, 0, tflops
            
        except Exception as e:
            print(f"Error in FusedMoE benchmark: {e}")
            return float('inf'), 0, 0
    
    def bench_ep_moe(self, bsz: int, seq_len: int) -> tuple:
        """
        Benchmark EPMoE implementation - Expert Parallelism for MoE.
        
        Args:
            bsz: Batch size
            seq_len: Sequence length
            
        Returns:
            Tuple of (latency_ms, throughput_tokens_per_s, throughput_tflops)
        """
        try:
            from sglang.srt.layers.moe.ep_moe.layer import EPMoE
            
            # Initialize the MoE model
            experts = EPMoE(
                num_experts=self.num_experts,
                top_k=self.top_k + 1,  # EPMoE uses an extra slot
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                renormalize=self.renormalize,
                quant_config=self.quant_config,
                use_grouped_topk=self.use_grouped_topk,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                tp_size=self.mp_size,
            ).to(self.device)
            
            # Prepare inputs - note that EP parallelism requires more tokens
            hidden_states = torch.randn(bsz * seq_len, self.hidden_size, dtype=torch.half, device=self.device)
            router_logits = torch.randn(bsz * seq_len, self.num_experts, dtype=torch.half, device=self.device)
            
            def run_moe():
                return experts(hidden_states=hidden_states, router_logits=router_logits)
            # print(run_moe().shape)
            # Benchmark run
            ms = do_bench(run_moe, warmup=self.warmup_iters, rep=self.benchmark_iters)
            
            # Calculate TFLOPS
            tflops = self._calculate_tflops(bsz, seq_len, ms)
            
            return ms, 0, tflops
            
        except Exception as e:
            print(f"Error in EPMoE benchmark: {e}")
            return float('inf'), 0, 0
         
    def bench_normal_deepep_moe(self, bsz: int, seq_len: int) -> tuple:
        """
        Benchmark Normal DeepEPMoE implementation - Deep Expert Parallelism in normal mode.
        
        Args:
            bsz: Batch size
            seq_len: Sequence length
            
        Returns:
            Tuple of (latency_ms, throughput_tokens_per_s, throughput_tflops)
        """
        try:
            from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
            from sglang.srt.utils import DeepEPMode
            from sglang.srt.model_executor.forward_batch_info import ForwardMode
            
            # Initialize the MoE model with normal mode
            experts = DeepEPMoE(
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
                deepep_mode=DeepEPMode.normal
            ).to(self.device)
            
            # Prepare inputs - FP8 quantized hidden states
            hidden_states_0 = torch.randn(bsz * seq_len, self.hidden_size, dtype=torch.half, device=self.device).to(torch.float8_e4m3fn).contiguous()
            hidden_states_1 = torch.randn(bsz * seq_len, self.hidden_size // 128, dtype=torch.float32, device=self.device).contiguous()
            
            # Create topk indices (using a simple pattern for benchmarking)
            topk_idx = torch.zeros((bsz * seq_len, self.top_k), dtype=torch.int, device=self.device)
            for i in range(self.top_k):
                topk_idx[:, i] = i
                
            # Create random weights for the selected experts
            topk_weights = torch.randn((bsz * seq_len, self.top_k), dtype=torch.float32, device=self.device)
            
            # Set up tokens per expert distribution (simplified for benchmark)
            
            tokens_per_expert = [max(128, bsz * seq_len)] * 2 + [0] * 14
            
            def run_moe():
                return experts(
                    hidden_states=(hidden_states_0, hidden_states_1),
                    topk_idx=topk_idx,
                    topk_weights=topk_weights,
                    reorder_topk_ids=None,
                    seg_indptr=None,
                    masked_m=None,
                    expected_m=None,
                    num_recv_tokens_per_expert=tokens_per_expert,
                    forward_mode=ForwardMode.EXTEND,
                )
                
            # Benchmark run
            ms = do_bench(run_moe, warmup=self.warmup_iters, rep=self.benchmark_iters)
            
            # Calculate TFLOPS
            tflops = self._calculate_tflops(bsz, seq_len, ms)
            
            return ms, 0, tflops
            
        except Exception as e:
            print(f"Error in DeepEPMoE Normal benchmark: {e}")
            return float('inf'), 0, 0
    
    def bench_low_lat_deepep_moe(self, bsz: int, seq_len: int) -> tuple:
        """
        Benchmark Low Latency DeepEPMoE implementation - optimized for lower latency.
        
        Args:
            bsz: Batch size
            seq_len: Sequence length
            
        Returns:
            Tuple of (latency_ms, throughput_tokens_per_s, throughput_tflops)
        """
        if bsz * seq_len > 128:
            return float('inf'), 0, 0  # Skip if too small input size
        try:
            from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
            from sglang.srt.utils import DeepEPMode
            from sglang.srt.model_executor.forward_batch_info import ForwardMode
            
            # Initialize the MoE model with low latency mode
            experts = DeepEPMoE(
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
                deepep_mode=DeepEPMode.low_latency
            ).to(self.device)
            
            # Prepare specialized inputs for low latency mode
            hidden_states_0 = torch.randn(
                self.num_experts // self.mp_size, self.intermediate_size, self.hidden_size, 
                dtype=torch.half, device=self.device
            ).to(torch.float8_e4m3fn)
            
            hidden_states_1 = torch.randn(
                self.num_experts // self.mp_size, self.intermediate_size, self.hidden_size // 128, 
                dtype=torch.float32, device=self.device
            )
            
            # Set up token mask for experts
            masked_m = torch.zeros((self.num_experts // self.mp_size), dtype=torch.int32, device=self.device)
            for i in range(self.top_k):
                masked_m[i] = bsz
            
            def run_moe():
                return experts(
                    hidden_states=(hidden_states_0, hidden_states_1),
                    topk_idx=None,
                    topk_weights=None,
                    reorder_topk_ids=None,
                    seg_indptr=None,
                    masked_m=masked_m,
                    expected_m=1,
                    num_recv_tokens_per_expert=None,
                    forward_mode=ForwardMode.EXTEND,
                )
                
            # Benchmark run
            ms = do_bench(run_moe, warmup=self.warmup_iters, rep=self.benchmark_iters)
            
            # Calculate TFLOPS
            tflops = self._calculate_tflops(bsz, seq_len, ms)
            
            return ms, 0.0, tflops
            
        except Exception as e:
            print(f"Error in DeepEPMoE Low Latency benchmark: {e}")
            return float('inf'), 0, 0
    
    def _calculate_tflops(self, bsz: int, seq_len: int, ms: float) -> float:
        """
        Helper method to calculate TFLOPS for MoE operations.
        
        Args:
            bsz: Batch size
            seq_len: Sequence length
            ms: Latency in milliseconds
            
        Returns:
            Throughput in TFLOPS
        """
        # Formula accounts for matrix multiplications in MoE forward pass
        # FLOPs = 2 * topk * tokens * hidden_size * (intermediate_size + intermediate_size/2) / mp_size
        return 2 * self.top_k * bsz * seq_len * self.hidden_size * (
            self.intermediate_size // self.mp_size + self.intermediate_size // 2 // self.mp_size
        ) * 1e-12 / (ms * 1e-3)