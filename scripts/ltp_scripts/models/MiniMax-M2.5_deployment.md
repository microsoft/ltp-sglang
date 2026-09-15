# MiniMax-M2.5 Deployment Guide (BF16, H100/H200)

## Overview

[MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) is a powerful MoE language model developed by MiniMax, built for real-world productivity with state-of-the-art performance across coding, reasoning, agentic tasks, and tool use. It achieves strong results on SWE-bench, AIME25, BrowseComp, and Terminal Bench 2.

This guide covers the **BF16 (full-precision)** deployment of MiniMax-M2.5 on **NVIDIA H100 and H200** GPUs using SGLang.

> **Note:** MiniMax-M2.5 supports both 4-GPU and 8-GPU single-node deployment on H100 and H200. Requires SGLang **≥ 0.5.9**.
>
> | Hardware | TP Size | EP Size | Min GPUs |
> |----------|---------|---------|----------|
> | H100 / H200 | tp=4  | ep=4  | 4        |
> | H100 / H200 | tp=8  | ep=8  | 8        |

---

## Docker Image

Use the standard SGLang image:

```bash
docker pull lmsysorg/sglang:v0.5.9-cu124
```

---

## Deployment

### Single Node

#### 4-GPU (tp=4)

```bash
python -m sglang.launch_server --model MiniMaxAI/MiniMax-M2.5 --tp 4 --ep 4 --trust-remote-code --reasoning-parser minimax-append-think --tool-call-parser minimax-m2 --mem-fraction-static 0.85 --host 0.0.0.0 --port 30000
```

#### 8-GPU (tp=8)

```bash
python -m sglang.launch_server --model MiniMaxAI/MiniMax-M2.5 --tp 8 --ep 8 --trust-remote-code --reasoning-parser minimax-append-think --tool-call-parser minimax-m2 --mem-fraction-static 0.85 --host 0.0.0.0 --port 30000
```

### Multi-Node

For clusters of 8-GPU nodes, run one command per node. Node 0 is the master and serves the API; worker nodes only participate in computation. Replace `<MASTER_IP>` with the IP of node 0.

#### 2 Nodes × 8 GPUs (tp=16, ep=16)

**Node 0 (master):**
```bash
python -m sglang.launch_server --model MiniMaxAI/MiniMax-M2.5 --tp 16 --ep 16 --nnodes 2 --node-rank 0 --dist-init-addr <MASTER_IP>:50000 --trust-remote-code --reasoning-parser minimax-append-think --tool-call-parser minimax-m2 --mem-fraction-static 0.85 --host 0.0.0.0 --port 30000
```

**Node 1:**
```bash
python -m sglang.launch_server --model MiniMaxAI/MiniMax-M2.5 --tp 16 --ep 16 --nnodes 2 --node-rank 1 --dist-init-addr <MASTER_IP>:50000 --trust-remote-code --reasoning-parser minimax-append-think --tool-call-parser minimax-m2 --mem-fraction-static 0.85
```

#### 4 Nodes × 8 GPUs (tp=32, ep=32)

**Node 0 (master):**
```bash
python -m sglang.launch_server --model MiniMaxAI/MiniMax-M2.5 --tp 32 --ep 32 --nnodes 4 --node-rank 0 --dist-init-addr <MASTER_IP>:50000 --trust-remote-code --reasoning-parser minimax-append-think --tool-call-parser minimax-m2 --mem-fraction-static 0.85 --host 0.0.0.0 --port 30000
```

**Nodes 1–3** (set `--node-rank` to 1, 2, or 3 on each):
```bash
python -m sglang.launch_server --model MiniMaxAI/MiniMax-M2.5 --tp 32 --ep 32 --nnodes 4 --node-rank <N> --dist-init-addr <MASTER_IP>:50000 --trust-remote-code --reasoning-parser minimax-append-think --tool-call-parser minimax-m2 --mem-fraction-static 0.85
```

---

## Key Launch Arguments

| Argument | Description |
|---|---|
| `--tp` | Tensor parallelism size (total GPUs across all nodes) |
| `--ep` | Expert parallelism size; set equal to `--tp` for MoE models |
| `--trust-remote-code` | Required for MiniMax model loading |
| `--reasoning-parser minimax-append-think` | Separates `<think>...</think>` content from the response |
| `--tool-call-parser minimax-m2` | Enables tool/function call parsing |
| `--mem-fraction-static 0.85` | Fraction of GPU memory reserved for static weights |
| `--nnodes` | Number of nodes (multi-node only) |
| `--node-rank` | Rank of the current node, 0 = master (multi-node only) |
| `--dist-init-addr` | `<master_ip>:<port>` for distributed coordination (multi-node only) |

---

## References

- [SGLang MiniMax-M2.5 Cookbook](https://cookbook.sglang.io/autoregressive/MiniMax/MiniMax-M2.5)
- [MiniMax-M2.5 on HuggingFace](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
