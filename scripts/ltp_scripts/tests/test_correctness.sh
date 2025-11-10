#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -euo pipefail

export TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-/mnt/output}"

modules=("attention" "logits" "rmsnorm" "rope" "token_embedding")
for module in "${modules[@]}"; do
    echo "*** Test correctness for $module ***"
    pytest -s "$(dirname $0)/../../../test/srt/numerical_tests/comparison_tests/benchmark/test_bench_$module.py"
    TEST_BENCHMARK_DIR="$TEST_OUTPUT_DIR/bench/$module" pytest -s "$(dirname $0)/../../../test/srt/numerical_tests/comparison_tests/test_${module}_comparison.py"
done

for f in $TEST_OUTPUT_DIR/*-compare/*/*.json; do
    printf '\n*** %s ***\n' "$f"
    jq . "$f"
done
