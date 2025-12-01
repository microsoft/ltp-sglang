#!/bin/bash

DEST_ROOT="/sgl-workspace/ltp-sglang/benchmark/sigma_v2/model_conf"

MODELS=(
  #"meta-llama/Meta-Llama-3.1-8B-Instruct"
  #"meta-llama/Meta-Llama-3.1-70B-Instruct"
  #"meta-llama/Llama-3.1-405B-Instruct"
  #"Qwen/Qwen3-4B"
  #"Qwen/Qwen3-32B"
  #"deepseek-ai/DeepSeek-V2-Lite"
  #"Qwen/Qwen3-30B-A3B"
  #"Qwen/Qwen3-235B-A22B"
  "moonshotai/Kimi-K2-Instruct"
  #"deepseek-ai/DeepSeek-R1"
  #"deepseek-ai/deepseek-vl2-tiny"
  "deepseek-ai/deepseek-vl2"
  "Qwen/Qwen2.5-VL-7B-Instruct"
)

for MODEL in "${MODELS[@]}"; do
    echo "=============================="
    echo "Running benchmark for $MODEL"
    echo "=============================="

    # 1. Run benchmark
    # Construct the command
    CMD="python -m sglang.bench_offline_throughput \
        --model-path \"$MODEL\" \
        --dataset-name random \
        --random-input 1 \
        --random-output 1 \
        --trust-remote-code \
        --load-format dummy \
        --num-prompts 1 \
        --tp-size 2 --ep-size 2"

    # Echo the command before running
    echo "Running command:"
    echo "$CMD"

    # Execute the command
    eval "$CMD"

    # 2. Convert model name to HF cache folder format
    CACHE_NAME=$(echo "$MODEL" | sed 's|/|--|g')
    CACHE_PATH="$HOME/.cache/huggingface/hub/models--$CACHE_NAME"

    # 3. Find snapshot directory inside cache
    SNAPSHOT=$(ls -d $CACHE_PATH/snapshots/* 2>/dev/null | head -n 1)

    if [[ -z "$SNAPSHOT" ]]; then
        echo "❌ ERROR: No snapshot found for $MODEL"
        continue
    fi

    echo "Snapshot found: $SNAPSHOT"

    # 4. Destination folder
    MODEL_BASENAME=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
    DEST="$DEST_ROOT/$MODEL_BASENAME"

    mkdir -p "$DEST"
    find "$DEST" -type l -delete
    find "$DEST" -type f -delete

    echo "Copying snapshot to: $DEST"
    cp -aL "${SNAPSHOT}/." "$DEST"

    # 5. Rename config.json → config_full.json
    if [[ -f "$DEST/config.json" ]]; then
        echo "Renaming config.json → config_full.json"
        mv "$DEST/config.json" "$DEST/config_full.json"
    else
        echo "⚠️ No config.json found in $DEST"
    fi

    echo "✅ Finished processing $MODEL"
    echo
done

