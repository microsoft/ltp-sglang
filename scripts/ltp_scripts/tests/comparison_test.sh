#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script is used to run the comparison test based on the provided Benchmark dataset.

# Usage: ./comparison_test.sh <dataset_path> <output_path>

dataset_path=$1
if [ -z "$dataset_path" ] ; then
    echo "Usage: $0 <dataset_path> <output_path>"
    exit 1
fi

output_path=${2:-"./comparison_results"}
mkdir -p "$output_path"

# Run the comparison tests for each module
script_dir=$(dirname "$0")
test_path="$script_dir/../../../test/srt/numerical_tests/comparison_tests"

run_comparison_test() {
    local module_name=$1

    echo "Running comparison test for $module_name..."
    TEST_BENCHMARK_DIR="$dataset_path/$module_name" TEST_OUTPUT_DIR=$output_path pytest -s "$test_path/test_${module_name}_comparison.py"
}

run_comparison_test "attention"
run_comparison_test "moe"
run_comparison_test "logits"
run_comparison_test "rmsnorm"
run_comparison_test "rope"
run_comparison_test "token_embedding"
