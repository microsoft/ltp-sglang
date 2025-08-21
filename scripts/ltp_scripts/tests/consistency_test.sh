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
export CHECKPOINT_PATH="$REAL_WEIGHTS_PATH"
export LOG_DIR="$LOG_DIR"

# Run the consistency test
script_dir=$(dirname "$0")
pytest -s "$script_dir/../../../test/srt/numerical_tests/consistency_tests"
