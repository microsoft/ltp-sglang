# Kimi-K2.5 Deployment Guide (BF16, H200)

## Overview

[Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) is an open-source native multimodal agentic model by Moonshot AI, built through continual pretraining on approximately 15 trillion mixed visual and text tokens. It integrates vision and language understanding with advanced agentic capabilities, supporting both thinking (reasoning) and instant modes.

This guide covers the **BF16 (full-precision)** deployment of Kimi-K2.5 on **NVIDIA H200** GPUs using SGLang.

> **Note:** Kimi-K2.5 requires GPUs with ≥140GB memory each. **H100 (80GB/94GB) is not supported.** H200 (141GB) is the minimum supported NVIDIA GPU. Requires SGLang **≥ 0.5.9**.
>
> | Hardware | BF16 TP Size |
> |----------|-------------|
> | H200     | tp=8        |

---

## Docker Image

Use the standard SGLang image for NVIDIA Hopper GPUs:

```bash
docker pull lmsysorg/sglang:v0.5.9-cu124
```

---

## Deployment

### Single Node

For a single 8-GPU H200 node (tp=8):

```bash
python -m sglang.launch_server --model moonshotai/Kimi-K2.5 --tp 8 --trust-remote-code --reasoning-parser kimi_k2 --tool-call-parser kimi_k2 --host 0.0.0.0 --port 30000
```

### Multi-Node

If your cluster only has 8-GPU nodes and you want to serve multiple requests at higher throughput across nodes, or if your workload warrants a larger TP degree, you can span across nodes. Replace `<MASTER_IP>` with the IP of node 0.

#### H200 — 2 Nodes × 8 GPUs (tp=16)

**Node 0 (master):**
```bash
python -m sglang.launch_server --model moonshotai/Kimi-K2.5 --tp 16 --nnodes 2 --node-rank 0 --dist-init-addr <MASTER_IP>:50000 --trust-remote-code --reasoning-parser kimi_k2 --tool-call-parser kimi_k2 --host 0.0.0.0 --port 30000
```

**Node 1:**
```bash
python -m sglang.launch_server --model moonshotai/Kimi-K2.5 --tp 16 --nnodes 2 --node-rank 1 --dist-init-addr <MASTER_IP>:50000 --trust-remote-code --reasoning-parser kimi_k2 --tool-call-parser kimi_k2
```

#### H200 — 4 Nodes × 8 GPUs (tp=32)

**Node 0 (master):**
```bash
python -m sglang.launch_server --model moonshotai/Kimi-K2.5 --tp 32 --nnodes 4 --node-rank 0 --dist-init-addr <MASTER_IP>:50000 --trust-remote-code --reasoning-parser kimi_k2 --tool-call-parser kimi_k2 --host 0.0.0.0 --port 30000
```

**Nodes 1–3** (set `--node-rank` to 1, 2, or 3 on each):
```bash
python -m sglang.launch_server --model moonshotai/Kimi-K2.5 --tp 32 --nnodes 4 --node-rank <N> --dist-init-addr <MASTER_IP>:50000 --trust-remote-code --reasoning-parser kimi_k2 --tool-call-parser kimi_k2
```

---

## Key Launch Arguments

| Argument | Description |
|---|---|
| `--tp` | Tensor parallelism size (total GPUs across all nodes) |
| `--trust-remote-code` | Required for the `kimi_k2.5` model architecture |
| `--nnodes` | Number of nodes (multi-node only) |
| `--node-rank` | Rank of the current node, 0 = master (multi-node only) |
| `--dist-init-addr` | `<master_ip>:<port>` for distributed coordination (multi-node only) |
| `--reasoning-parser kimi_k2` | Enables structured extraction of thinking/reasoning content |
| `--tool-call-parser kimi_k2` | Enables tool/function call parsing |

---

## References

- [SGLang Kimi-K2.5 Cookbook](https://cookbook.sglang.io/autoregressive/Moonshotai/Kimi-K2.5)
- [Kimi-K2.5 on HuggingFace](https://huggingface.co/moonshotai/Kimi-K2.5)
