# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# !/bin/bash

# Benchmark generation script for numerical tests
# This script generates benchmark datasets for numerical comparison between SGLang and Transformers implementations.

REAL_WEIGHTS_PATH=$1
if [ -z "$REAL_WEIGHTS_PATH" ]; then
    echo "Usage: $0 <real_weights_path> [log_dir]"
    exit 1
fi
LOG_DIR=${2:-"benchmark"}
mkdir -p "$LOG_DIR"

# Set environment variables for the test
export CHECKPOINT_PATH="$REAL_WEIGHTS_PATH"
export LOG_DIR="$LOG_DIR"

# Run the benchmark generation test
script_dir=$(dirname "$0")
pytest -s "$script_dir/../../../test/srt/numerical_tests/comparison_tests/benchmark"
