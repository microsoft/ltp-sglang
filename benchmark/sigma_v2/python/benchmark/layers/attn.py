import torch
import torch.nn as nn
import sglang.srt.layers.quantization.base_config
from sglang.srt.layers.linear import QKVParallelLinear
import matplotlib.pyplot as plt
from flash_attn_interface import flash_attn_varlen_func

class Attn(nn.Module):
    def __init__(self, tp_size):
        super().__init__()
        self.qkv_proj = QKVParallelLinear(
            5120,
            128,
            64,
            8,
            bias=False,
            params_dtype=torch.half,
            quant_config=None,
            tp_rank=0,
            tp_size=tp_size,
            prefix="qkv",
        )
        self.num_heads = 64 // tp_size
        self.num_kv_heads = max(1, 8 // tp_size)
        self.tp_size = tp_size
        self.q_size = self.num_heads * 128
        self.kv_size = self.num_kv_heads * 128
        self.device = torch.device("cuda:0")
    
    def forward(self, x, seq_len):
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(seq_len, 64 // self.tp_size, 128)
        k = k.view(seq_len, max(1, 8 // self.tp_size), 128)
        v = v.view(seq_len, max(1, 8 // self.tp_size), 128)
        cu_seqlens_q = torch.arange(1, device=self.device, dtype=torch.int32) * seq_len
        cu_seqlens_k = torch.arange(1, device=self.device, dtype=torch.int32) * seq_len
        attn_output = flash_attn_varlen_func(
                q=q, 
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                causal=False,
        )
        return attn_output
    
tp_token_time = {}
for tp_size in [1, 8, 16]:
    tp_token_time[tp_size] = {}
    for tokens in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        attn = Attn(tp_size).cuda()
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
        tp_token_time[tp_size][tokens] = elapsed_time / 50
        print(f"TP Size: {tp_size}, Tokens: {tokens}, Time: {elapsed_time:.2f} ms")
        print(len(output_tensor))
print(tp_token_time)
fig = plt.figure(figsize=(10, 6))
for tp_size, times in tp_token_time.items():
    tokens = list(times.keys())
    elapsed_times = list(times.values())
    throughput = [tokens[i] / (elapsed_times[i] / 1000) for i in range(len(tokens))]
    plt.plot(tokens, throughput, label=f'TP Size {tp_size}', marker='o')
plt.xlabel('Number of Tokens')
plt.ylabel('Throughput (token/s)')
plt.legend()
plt.grid(True)
plt.savefig('tp_size_vs_time.pdf')