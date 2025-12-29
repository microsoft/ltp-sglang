# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# !/bin/bash

# Consistency test script for SGLang
# This script runs a series of tests to ensure that the SGLang module behaves consistently in the same environment.

REAL_WEIGHTS_PATH=$1
if [ -z "$REAL_WEIGHTS_PATH" ]; then
    echo "Usage: $0 <real_weights_path> [log_dir]"
    exit 1
fi
LOG_DIR=${2:-"consistency_test"}
mkdir -p "$LOG_DIR"

# Set environment variables for the test
export TEST_CHECKPOINT_DIR="$REAL_WEIGHTS_PATH"
export TEST_OUTPUT_DIR="$LOG_DIR"
# Disable AITER for consistency tests due to aiter did not support current moe module
export SGLANG_USE_AITER=false

# Run the consistency test
script_dir=$(dirname "$0")
pytest -s "$script_dir/../../../test/srt/numerical_tests/consistency_tests/test_attention.py"
pytest -s "$script_dir/../../../test/srt/numerical_tests/consistency_tests/test_logits.py"
pytest -s "$script_dir/../../../test/srt/numerical_tests/consistency_tests/test_moe.py"
pytest -s "$script_dir/../../../test/srt/numerical_tests/consistency_tests/test_rmsnorm.py"
pytest -s "$script_dir/../../../test/srt/numerical_tests/consistency_tests/test_rope.py"
pytest -s "$script_dir/../../../test/srt/numerical_tests/consistency_tests/test_token_embedding.py"
