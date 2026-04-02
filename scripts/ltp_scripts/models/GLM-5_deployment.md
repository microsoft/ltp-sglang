# GLM-5 Deployment Guide (BF16, H100/H200)

## Overview

[GLM-5](https://huggingface.co/zai-org/GLM-5) is the most powerful model in the GLM series by Zhipu AI, with 744B total parameters (40B active). It is designed for complex systems engineering and long-horizon agentic tasks, achieving best-in-class performance among open-source models on reasoning, coding, and agentic benchmarks.

This guide covers the **BF16 (full-precision)** deployment of GLM-5 on **NVIDIA H100 and H200** GPUs using SGLang.

> **Note:** BF16 requires 2× the GPU count compared to FP8 ([zai-org/GLM-5-FP8](https://huggingface.co/zai-org/GLM-5-FP8)).
>
> | Hardware | BF16 TP Size | Nodes (8 GPUs each) |
> |----------|-------------|---------------------|
> | H200     | tp=16       | 2 nodes             |
> | H100     | tp=32       | 4 nodes             |

---

## Docker Image

Use the dedicated Hopper image for H100/H200:

```bash
docker pull lmsysorg/sglang:glm5-hopper
```

---

## Deployment

### Single Node

For a single machine with sufficient GPUs (e.g., a 16-GPU H200 node or 32-GPU H100 node):

**H200 (tp=16):**
```bash
python -m sglang.launch_server --model zai-org/GLM-5 --tp 16 --reasoning-parser glm45 --tool-call-parser glm47 --mem-fraction-static 0.85 --host 0.0.0.0 --port 30000
```

**H100 (tp=32):**
```bash
python -m sglang.launch_server --model zai-org/GLM-5 --tp 32 --reasoning-parser glm45 --tool-call-parser glm47 --mem-fraction-static 0.85 --host 0.0.0.0 --port 30000
```

### Multi-Node

For clusters of 8-GPU nodes, run one command per node. Node 0 is the master and serves the API; worker nodes only participate in computation. Replace `<MASTER_IP>` with the IP of node 0.

#### H200 — 2 Nodes × 8 GPUs (tp=16)

**Node 0 (master):**
```bash
python -m sglang.launch_server --model zai-org/GLM-5 --tp 16 --nnodes 2 --node-rank 0 --dist-init-addr <MASTER_IP>:50000 --reasoning-parser glm45 --tool-call-parser glm47 --mem-fraction-static 0.85 --host 0.0.0.0 --port 30000
```

**Node 1:**
```bash
python -m sglang.launch_server --model zai-org/GLM-5 --tp 16 --nnodes 2 --node-rank 1 --dist-init-addr <MASTER_IP>:50000 --reasoning-parser glm45 --tool-call-parser glm47 --mem-fraction-static 0.85
```

#### H100 — 4 Nodes × 8 GPUs (tp=32)

**Node 0 (master):**
```bash
python -m sglang.launch_server --model zai-org/GLM-5 --tp 32 --nnodes 4 --node-rank 0 --dist-init-addr <MASTER_IP>:50000 --reasoning-parser glm45 --tool-call-parser glm47 --mem-fraction-static 0.85 --host 0.0.0.0 --port 30000
```

**Nodes 1–3** (set `--node-rank` to 1, 2, or 3 on each):
```bash
python -m sglang.launch_server --model zai-org/GLM-5 --tp 32 --nnodes 4 --node-rank <N> --dist-init-addr <MASTER_IP>:50000 --reasoning-parser glm45 --tool-call-parser glm47 --mem-fraction-static 0.85
```

---

## Key Launch Arguments

| Argument | Description |
|---|---|
| `--tp` | Tensor parallelism size (total GPUs across all nodes) |
| `--nnodes` | Number of nodes |
| `--node-rank` | Rank of the current node (0 = master) |
| `--dist-init-addr` | `<master_ip>:<port>` for distributed coordination |
| `--reasoning-parser glm45` | Enables structured extraction of thinking/reasoning content |
| `--tool-call-parser glm47` | Enables tool/function call parsing |
| `--mem-fraction-static 0.85` | Fraction of GPU memory reserved for static weights |

---

## References

- [SGLang GLM-5 Cookbook](https://cookbook.sglang.io/autoregressive/GLM/GLM-5)
- [GLM-5 on HuggingFace](https://huggingface.co/zai-org/GLM-5)
