#!/bin/bash
set -e

download_if_missing() {
    local dir="$1"
    shift
    mkdir -p "$dir"
    cd "$dir"
    for file_url in "$@"; do
        file="${file_url##*/}"
        [ -f "$file" ] || wget "$file_url"
    done
    touch config.json
    cd - > /dev/null
}

download_if_missing "model_conf/deepseek" \
    "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer.json" \
    "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/configuration_deepseek.py" \
    "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer_config.json"

download_if_missing "model_conf/qwen" \
    "https://huggingface.co/Qwen/Qwen3-235B-A22B/resolve/main/tokenizer.json" \
    "https://huggingface.co/Qwen/Qwen3-235B-A22B/resolve/main/tokenizer_config.json"

download_if_missing "model_conf/qwen_sigma" \
    "https://huggingface.co/Qwen/Qwen3-235B-A22B/resolve/main/tokenizer.json" \
    "https://huggingface.co/Qwen/Qwen3-235B-A22B/resolve/main/tokenizer_config.json"