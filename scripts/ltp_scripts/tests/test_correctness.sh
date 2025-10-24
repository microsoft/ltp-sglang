#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


export TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-/mnt/output}"

modules=("attention" "moe" "logits" "rmsnorm" "rope" "token_embedding")
for module in "${modules[@]}"; do
    echo "=== Test correctness for $fruit ==="
    pytest -s "$(dirname $0)/../../../test/srt/numerical_tests/comparison_tests/benchmark/test_bench_$module.py"
    TEST_BENCHMARK_DIR="$TEST_OUTPUT_DIR/bench/$module" pytest -s "$(dirname $0)/../../../test/srt/numerical_tests/comparison_tests/test_${module_name}_comparison.py"
done
