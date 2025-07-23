import torch
from triton.testing import do_bench
import matplotlib.pyplot as plt

from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from layer import LayerBenchmark

class AttentionBenchmark(LayerBenchmark):
    def __init__(
        self,
        name: str = "Attention",
        hidden_size: int = 5120,
        head_dim: int = 128,
        num_attention_heads: int = 64,
        num_kv_heads: int = 8,
        mp_size: int = 1,
    ):
        super().__init__(name)
        import mp_patch
        mp_patch.apply_hardcode_parallel(mp_size)
        import sglang.srt.layers.quantization.base_config
        from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.mp_size = mp_size
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_attention_heads,
            self.num_kv_heads,
            bias=False,
            params_dtype=torch.half,
            quant_config=None,
            tp_rank=0,
            tp_size=self.mp_size,
            prefix="qkv",
        ).to(self.device)
        self.o_proj = RowParallelLinear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=None,
            tp_rank=0,
            tp_size=self.mp_size,
            params_dtype=torch.half,
            reduce_results=False,
            prefix="o_proj",
        ).to(self.device)
        self.num_heads = self.num_attention_heads // self.mp_size
        self.num_kv_heads = max(1, self.num_kv_heads // self.mp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        
        self.benchmark_funcs = {
            "prefill_attn": self.bench_prefill_attn,
            "decode_attn": self.bench_decode_attn,
        }
        
    def bench_prefill_attn(self, bsz: int, seq_len: int) -> tuple:
        x = torch.randn(bsz, seq_len, self.hidden_size, dtype=torch.half, device=self.device)
        
        def attn(): 
            qkv, _ = self.qkv_proj(x)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
            k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            attn_output = flash_attn_with_kvcache(
                q=q, 
                k_cache=k,
                v_cache=v,
                causal=False,
            )
            attn_output = attn_output.view(bsz, seq_len, self.num_heads * self.head_dim)
            output, _ = self.o_proj(attn_output)
            return output
        
        output = attn()
        ms = do_bench(attn, warmup=self.warmup_iters, rep=self.benchmark_iters)
        io_gb_per_s = self._calculate_io_gb_per_s([x, self.qkv_proj.weight, output, self.o_proj.weight], ms)
        tflops = self._calculate_prefill_tflops(bsz, seq_len, ms)
        return ms, io_gb_per_s, tflops

    def bench_decode_attn(self, bsz: int, seq_len: int) -> tuple:
        x = torch.randn(bsz, 1, self.hidden_size, dtype=torch.half, device=self.device)
        key_cache = torch.randn(bsz, seq_len, self.num_kv_heads, self.head_dim, dtype=torch.half, device=self.device)
        value_cache = torch.randn(bsz, seq_len, self.num_kv_heads, self.head_dim, dtype=torch.half, device=self.device)
        def attn(): 
            qkv, _ = self.qkv_proj(x)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q = q.view(bsz, 1, self.num_heads, self.head_dim)
            k = k.view(bsz, 1, self.num_kv_heads, self.head_dim)
            v = v.view(bsz, 1, self.num_kv_heads, self.head_dim)
            k_cache = torch.cat([key_cache, k], dim=1)
            v_cache = torch.cat([value_cache, v], dim=1)
            attn_output = flash_attn_with_kvcache(
                q=q, 
                k_cache=k_cache,
                v_cache=v_cache,
                causal=False,
            )
            attn_output = attn_output.view(bsz, 1, self.num_heads * self.head_dim)
            output, _ = self.o_proj(attn_output)
            return output
        
        output = attn()
        ms = do_bench(attn, warmup=self.warmup_iters, rep=self.benchmark_iters)
        io_gb_per_s = self._calculate_io_gb_per_s([x, self.qkv_proj.weight, key_cache, value_cache, output, self.o_proj.weight], ms)
        tflops = self._calculate_decode_tflops(bsz, seq_len, ms)
        return ms, io_gb_per_s, tflops
     
    def _calculate_prefill_tflops(self, bsz: int, seq_len: int, ms: float) -> float:
        flops = 2 * bsz * seq_len * self.hidden_size * (self.q_size + self.kv_size * 2) \
            + bsz * 2 * seq_len * seq_len * self.num_heads * self.head_dim \
            + bsz * 2 * seq_len * self.hidden_size * self.hidden_size
        tflops = flops * 1e-12 / (ms * 1e-3)
        return tflops

    def _calculate_decode_tflops(self, bsz: int, seq_len: int, ms: float) -> float:
        flops = 2 * bsz * 1 * self.hidden_size * (self.q_size + self.kv_size * 2) \
            + bsz * 2 * 1 * seq_len * self.num_heads * self.head_dim \
            + bsz * 2 * 1 * self.hidden_size * self.hidden_size
        tflops = flops * 1e-12 / (ms * 1e-3)
        return tflops
    
    def _calculate_io_gb_per_s(self, tensors, ms):
        total_size = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        io_gb_per_s = (total_size / 1e9) / (ms * 1e-3)
        return io_gb_per_s

if __name__ == "__main__":
    tp_token_time = {}
    for mp_size in [1, 8, 16]:
        tp_token_time[mp_size] = {}
        for tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
            attn = AttentionBenchmark(mp_size).cuda()
            input_tensor = torch.randn(tokens, 5120, dtype=torch.half).cuda()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for i in range(100):
                if i == 50:
                    start_event.record()
                output_tensor = attn(input_tensor, tokens)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            tp_token_time[mp_size][tokens] = elapsed_time / 50
            print(f"TP Size: {mp_size}, Tokens: {tokens}, Time: {elapsed_time:.2f} ms")
            print(len(output_tensor))
    print(tp_token_time)
    fig = plt.figure(figsize=(10, 6))
    for mp_size, times in tp_token_time.items():
        tokens = list(times.keys())
        elapsed_times = list(times.values())
        throughput = [tokens[i] / (elapsed_times[i] / 1000) for i in range(len(tokens))]
        plt.plot(tokens, throughput, label=f'TP Size {mp_size}', marker='o')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Throughput (token/s)')
    plt.legend()
    plt.grid(True)
    plt.savefig('mp_size_vs_time.pdf')