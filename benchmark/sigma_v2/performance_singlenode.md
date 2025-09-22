# Single-node Performance
We compare Sigma V2's inference performance on vLLM (458e74eb907f96069e6d8a4f3c9f457001fef2ea) and SGLang (v0.4.10.post2).

## Long-context Performance (Sep 22, 2025)
We evaluate a specific input size: 500 prompt length with 10,000 generation length on H200.
### Key Takeaway
* Deployment Decision:
    * Low-latency: **TP8EP8** of SGLang
    * High-throughput: **DP8EP8** of SGLang
* SGLang outperforms vLLM in all batch sizes we evaluated for all deployment decisions in terms of E2E latency and throughput.
* Though SGLang has better performance, prefill and decode MFU are still extremely low.

### Why SGLang is better?
1. **Unified CUDA Graph Execution:**
    SGLang optimizes the decode phase by encapsulating the entire model forward pass into a single CUDA graph. In contrast, vLLM creates a separate CUDA graph for each model layer (e.g., 63 graphs for a 63-layer model). This design in vLLM introduces significant CPU overhead for each graph launch, resulting in increased GPU idle time and reduced efficiency. SGLang’s unified approach minimizes CPU launches and maximizes GPU utilization.

2. **Efficient Collective Communication:**
    SGLang streamlines AllReduce and AllGather operations by batching data transfers between Attention and MoE into a single collective operation per batch. This larger, consolidated transfer improves bandwidth utilization and overall efficiency. vLLM, however, performs multiple smaller transfers proportional to batch size, leading to higher communication overhead and lower throughput.

While other factors exist, their impact is relatively minor compared to the two main differences above.

### Additional Observations
1. **Torch Compile Integration:**
    vLLM enables Torch Compile by default, whereas SGLang requires explicit activation via arguments. Torch Compile provides noticeable performance improvements for SGLang only at batch size 1; for larger batch sizes, its effect is negligible. Kernel-level execution times are generally better in vLLM due to Torch Compile.

2. **DP Attention Implementation:**
    vLLM’s DP Attention is implemented with a more complex approach, making the process of launching DP tasks more cumbersome compared to SGLang. SGLang, overall, has a clearner and better-designed architecture.

3. **KV Cache Available Memory**
    vLLM and SGLang performs an analysis of how much memory can be reserved for KV cache, indicating the maximum generation length. Based on the logs, SGLang seems to allow a bit larger KV cache than vLLM.

<details>
<summary>Detailed Performance</summary>

| Framework           | Parallelism | GPU | BS  | TTFT (ms) | TPOT (ms) | E2E (s) | Token/s | Token/GPU/s | Prefill MFU | Decode MFU | Note           |
|---------------------|-------------|-----|-----|-----------|-----------|---------|---------|--------------|--------------|-------------|----------------|
| SGLang      | TP          | 8   | 1   | 114.86    | 7.21      | 72.18   | 145.47  | 18.18        | 2.64         | 0.08        | Torch Compile  |
|                     |             | 8   | 2   | 129.22    | 82.74     | 827.45  | 25.38   | 3.17         | 4.69         | 0.01        |                |
|                     |             | 8   | 4   | 156.98    | 84.16     | 841.67  | 49.90   | 6.24         | 7.72         | 0.03        |                |
|                     |             | 8   | 8   | 220.28    | 86.47     | 864.83  | 97.13   | 12.14        | 11.01        | 0.06        |                |
|                     |             | 8   | 16  | 346.04    | 90.70     | 907.26  | 185.17  | 23.15        | 14.01        | 0.11        |                |
|                     |             | 8   | 32  | 599.39    | 93.37     | 934.21  | 359.66  | 44.96        | 16.18        | 0.21        |                |
|                     |             | 8   | 64  | 1100.35   | 96.90     | 970.03  | 692.76  | 86.60        | 17.63        | 0.40        |                |
|                     |             | 8   | 128 | 2202.64   | 101.63    | 1018.45 | 1319.66 | 164.96       | 17.61        | 0.76        |                |
|                     |             | 8   | 256 | 4407.19   | 106.72    | 1071.48 | 2508.68 | 313.59       | 17.61        | 1.45        |                |
|                     |             | 8   | 512 | 8815.79   | 118.57    | 1194.42 | 4500.94 | 562.62       | 17.60        | 2.62        |                |
|                     | TP&EP       | 8   | 1   | 58.61     | 9.68      | 96.85   | 108.42  | 13.55        | 5.17         | 0.06        | Torch Compile  |
|                     |             | 8   | 2   | 71.25     | 10.74     | 107.46  | 195.42  | 24.43        | 8.51         | 0.11        |                |
|                     |             | 8   | 4   | 90.94     | 12.72     | 127.28  | 329.99  | 41.25        | 13.33        | 0.19        |                |
|                     |             | 8   | 8   | 158.57    | 14.31     | 143.24  | 586.41  | 73.30        | 15.29        | 0.34        |                |
|                     |             | 8   | 16  | 291.19    | 22.55     | 225.77  | 744.12  | 93.02        | 16.65        | 0.43        |                |
|                     |             | 8   | 32  | 560.15    | 27.56     | 276.13  | 1216.81 | 152.10       | 17.31        | 0.70        |                |
|                     |             | 8   | 64  | 1109.21   | 29.06     | 291.69  | 2303.79 | 287.97       | 17.49        | 1.33        |                |
|                     |             | 8   | 128 | 2221.99   | 30.77     | 309.94  | 4336.32 | 542.04       | 17.46        | 2.52        |                |
|                     |             | 8   | 256 | 4451.58   | 35.84     | 362.84  | 7408.27 | 926.03       | 17.43        | 4.33        |                |
|                     |             | 8   | 512 | 8867.42   | 55.27     | 561.53  | 9573.77 | 1196.72      | 17.50        | 5.62        |                |
|                     | DP&EP       | 8   | 8   | 150.83    | 18.63     | 186.43  | 450.57  | 56.32        | 16.08        | 0.26        |                |
|                     |             | 8   | 16  | 277.70    | 22.22     | 222.46  | 755.21  | 94.40        | 17.46        | 0.44        |                |
|                     |             | 8   | 32  | 524.19    | 24.39     | 244.40  | 1374.80 | 171.85       | 18.50        | 0.80        |                |
|                     |             | 8   | 64  | 1032.01   | 37.66     | 377.59  | 1779.69 | 222.46       | 18.80        | 1.03        |                |
|                     |             | 8   | 128 | 2074.93   | 38.37     | 385.74  | 3484.24 | 435.53       | 18.70        | 2.02        |                |
|                     |             | 8   | 256 | 4141.79   | 41.64     | 420.50  | 6392.39 | 799.05       | 18.73        | 3.73        |                |
|                     |             | 8   | 512 | 8252.76   | 51.57     | 523.90  | 10261.48| 1282.68      | 18.80        | 6.02        |                |
| vLLM     | TP          | 8   | 1   | 53.25     | 10.34     | 103.41  | 101.53  | 12.69        | 5.69         | 0.06        |                |
|                     |             | 8   | 2   | 91.80     | 12.05     | 120.57  | 174.17  | 21.77        | 6.60         | 0.10        |                |
|                     |             | 8   | 4   | 129.55    | 14.55     | 145.65  | 288.37  | 36.05        | 9.36         | 0.17        |                |
|                     |             | 8   | 8   | 216.38    | 19.53     | 195.47  | 429.72  | 53.72        | 11.21        | 0.25        |                |
|                     |             | 8   | 16  | 369.15    | 24.33     | 243.64  | 689.55  | 86.19        | 13.14        | 0.40        |                |
|                     |             | 8   | 32  | 687.82    | 28.70     | 287.64  | 1168.14 | 146.02       | 14.10        | 0.68        |                |
|                     |             | 8   | 64  | 1331.80   | 33.00     | 331.34  | 2028.13 | 253.52       | 14.56        | 1.18        |                |
|                     |             | 8   | 128 | 2621.42   | 41.75     | 420.12  | 3199.07 | 399.88       | 14.80        | 1.86        |                |
|                     |             | 8   | 256 | 5219.04   | 46.06     | 465.78  | 5770.95 | 721.37       | 14.87        | 3.37        |                |
|                     | TP&EP       | 8   | 1   | 48.05     | 10.78     | 107.80  | 97.40   | 12.17        | 6.31         | 0.06        |                |
|                     |             | 8   | 2   | 74.11     | 13.19     | 131.95  | 159.15  | 19.89        | 8.18         | 0.09        |                |
|                     |             | 8   | 4   | 110.30    | 13.87     | 138.82  | 302.55  | 37.82        | 10.99        | 0.17        |                |
|                     |             | 8   | 8   | 183.57    | 19.30     | 193.18  | 434.83  | 54.35        | 13.21        | 0.25        |                |
|                     |             | 8   | 16  | 321.73    | 26.59     | 266.18  | 631.15  | 78.89        | 15.07        | 0.36        |                |
|                     |             | 8   | 32  | 606.03    | 29.45     | 295.03  | 1138.85 | 142.36       | 16.00        | 0.66        |                |
|                     |             | 8   | 64  | 1203.42   | 30.59     | 307.11  | 2188.14 | 273.52       | 16.12        | 1.27        |                |
|                     |             | 8   | 128 | 2385.54   | 33.42     | 336.59  | 3992.99 | 499.12       | 16.26        | 2.32        |                |
|                     |             | 8   | 256 | 4739.73   | 38.65     | 391.19  | 6871.34 | 858.92       | 16.37        | 4.02        |                |
|                     | DP&EP       | 8   | 8   | 207.81    | 48.06     | 480.75  | 174.73  | 21.84        | 11.67        | 0.10        |                |
|                     |             | 8   | 16  | 337.53    | 58.86     | 588.88  | 285.29  | 35.66        | 14.37        | 0.16        |                |
|                     |             | 8   | 32  | 629.65    | 58.36     | 584.17  | 575.18  | 71.90        | 15.40        | 0.33        |                |
|                     |             | 8   | 64  | 1181.93   | 65.02     | 651.34  | 1031.72 | 128.97       | 16.41        | 0.60        |                |
|                     |             | 8   | 128 | 2284.01   | 76.62     | 768.40  | 1749.08 | 218.64       | 16.99        | 1.01        |                |
|                     |             | 8   | 256 | 4483.56   | 84.69     | 851.33  | 3157.42 | 394.68       | 17.31        | 1.83        |                |
|                     |             | 8   | 512 | 8970.09   | 92.20     | 930.92  | 5774.93 | 721.87       | 17.30        | 3.37        |                |

</details>

### Notes
We make some necessary modifications to the framework during evaluation to get stable results. SGLang's modifications are all updated in this repository.

## To be updated by Sep 26.
TODO: Jiamin will add more in-depth analysis of why SGLang MFU is still low.
