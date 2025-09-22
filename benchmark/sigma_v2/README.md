# Overview
This benchmark suite is designed to measure the inference latency of LLMs across different input sizes and deployment configurations. Key features include:

- **Flexible Input Sizes:** Benchmark multiple batch sizes and sequence lengths in a single run.
- **Focused Measurement:** Only the model's forward pass is timed, excluding preprocessing and postprocessing.
- **Streamlined Workflow:** Avoids complex command-line arguments—launch the inference server and send requests with simple, concise commands.

Tested on models such as GPT-3, Grok, Qwen, DeepSeek, and Sigma using H200 nodes.


# Quick Start
In quick start, we use existing YAML configurations under `conf` folder.
## Model Setup

1. **Create a model directory:**  
    Under the `model_conf` folder, create a subfolder for your model (e.g., `model_conf/qwen` for Qwen).

2. **Add required files:**  
    Place the following files in your model's folder:
    - `config.json` (model configuration)
    - `tokenizer.json` (tokenizer data)
    - `tokenizer_config.json` (tokenizer settings)

3. **Organize configuration files:**  
    - Rename `config.json` to `config_full.json` to represent the full model configuration.
    - To benchmark a smaller architecture (e.g., with 2 layers), duplicate `config_full.json`, modify the layer count to 2, and save as `config_2l.json`.
    - This approach allows you to easily benchmark different model variants (e.g., varying layer numbers or expert scales).

4. **Customize batch size and sequence length:**  
    - Edit `scripts/bsz_seq.csv` to specify your desired batch sizes and sequence lengths.
    - Use the format `{bsz1,bsz2};{seq1,seq2}` (e.g., `1,2,4;128,256,512`).
    - The benchmark will iterate over all combinations of batch sizes and sequence lengths.


## Running the Benchmark

### Single Node
```bash
# Benchmark the full Qwen model with Tensor Parallelism (TP) and Expert Parallelism (EP)
bash scripts/run.sh --model qwen --mconf full --deploy tp8ep8

# Benchmark a 2-layer Qwen model with Data Parallelism (DP), TP, and EP
bash scripts/run.sh --model qwen --mconf 2l --deploy tp8ep8dp8

# Benchmark and save results to a CSV file
bash scripts/run.sh --model qwen --mconf 2l --deploy tp8ep8 --save

# Profile both prefill and decode stages
bash scripts/run.sh --model qwen --mconf 2l --deploy tp8ep8 --profile both

# Profile only the decode stage
bash scripts/run.sh --model qwen --mconf 2l --deploy tp8ep8 --profile decode

# Use a custom base.yaml configuration
bash scripts/run.sh --model qwen --base {custom_base} --mconf 2l --deploy tp8ep8 --profile decode
```

### Multi-Node
```bash
# On the master node 
bash scripts/run.sh --model qwen --mconf full --deploy tp16ep16 --ip {master_ip}

# On a worker node 
bash scripts/run.sh --model qwen --mconf full --deploy tp16ep16 --ip {master_ip} --rank 1
```

### Results

Benchmark outputs are organized as follows:

- **`results/raw_output`**: Contains raw log files from each run.
- **`results/csv`**: Stores benchmark results in CSV format when `--save` is specified.
- **`results/{model}_profile`**: Includes NSight profiling report and SQLite files for in-depth analysis.

# Advanced Usage
To tailor the benchmark to your needs, organize and modify the following directories and files:

1. **`conf`**: Contains YAML configuration files that define SGLang server arguments and deployment strategies.
    - **Common required configs:** duplicate an existing YAML file and modify the values of the fields as needed.
      - **`base.yaml`**: basic server parameters 
      - Why we set `base.yaml` like this?
        <details>
        <summary>Click to expand example <code>base.yaml</code> configuration</summary>

        ```yaml
        model: ./model_conf/qwen_sigma
        load_format: dummy # Use dummy weights; actual model weights are not required for benchmarking.
        trust_remote_code: true # Permit loading custom model code from remote sources.

        log_level: debug # Set logging to debug for detailed output during benchmarking.

        disable_radix_cache: true # Disable radix cache to ensure all computations are performed; avoids skipping due to request similarity, ensuring stable results.
        disable_overlap_schedule: true # Disable overlapping schedules; only one batch per input size is run, so overlapping is unnecessary.
        disable_chunked_prefix_cache: true # Disable chunked prefix cache for consistent benchmarking across input sizes.
        enable_torch_compile: false # Enable Torch compile for performance optimization when batch size is 1; keep off for general benchmarking.

        mem_fraction_static: 0.5 # Set static memory fraction to 0.5 (default is 0.9 in SGLang for long-context); lower value reduces memory usage.

        max_running_requests: 16384 # Maximum concurrent requests; increase for benchmarking large-scale request scenarios.
        max_prefill_tokens: 1048576 # Maximum tokens for prefill in a single batch; raise for benchmarking very large input sequences.
        chunked_prefill_size: -1 # No chunking of prefill size; iterate from smallest to largest sequence length without splitting.

        max_tokens_generated: 2 # Limit generation to 2 tokens; only one prefill and one decode run to minimize benchmarking time.

        enable_benchmark: true # Enable benchmarking mode; runs 50 iterations to calculate average latency.
        enforce_batching: false # Enforce exact batch sizes for benchmarking (feature not yet merged in ltp-sglang).

        profile_phase: null # No profiling phase specified; set to 'prefill', 'decode', or 'both' to enable profiling.
        ```

        </details>
      - **`deploy/`**: templates for different parallelism deployement (e.g., TP, TP+EP, DP+EP).
      - **`backend/`**: backend options for Attention (e.g.FlashAttention3, FlashInfer)
        * Note: Future SGLang releases will support backend options for the MoE layer.*
    - **Optional configs:**  
      - **`optional/`**: offers the flexibility to use other [SGLang-supported arguments](https://docs.sglang.ai/advanced_features/server_arguments.html). I listed out a few in `default.yaml`. Add more if you want.
    ```bash
    # Assume you create YAML files like this
    # base: base_torchcompile
    # deploy: tp4ep4
    # backend: flashinfer
    # optional: nccl_ar
    bash scripts/run.sh --model sigma --base base_torchcompile --deploy tp4ep4 --backend flashinfer --optional nccl_ar
    ```
2. **`model_conf`**: Stores model-specific configuration files.
    - Each model has its own subfolder containing at least:
        - `config.json` (model configuration)
        - `tokenizer.json` (tokenizer data)
        - `tokenizer_config.json` (tokenizer settings)
    - To benchmark different model variants (e.g., fewer layers), create additional config files such as `config_full.json` or `config_2l.json`. Select the desired config when running the benchmark scripts.

3. **`python/benchmark`**: Contains the main Python script for running benchmarks.

4. **`python/tools`**: Includes utility scripts for counting model parameters and analyzing benchmark results, including MFU (Model FLOPs Utilization) computation.

5. **`scripts/run.sh`**: Bash script to launch the benchmark with various options.

6. **`scripts/bsz_seq.csv`**: Specifies batch sizes and sequence lengths for input combinations.

By customizing these files and folders, you can easily adapt the benchmark to different models, deployment strategies, and input sizes.
