# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import torch

# Set the distributed environment variables
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

# Set the checkpoint path and log directory
CHECKPOINT_PATH = os.getenv("TEST_CHECKPOINT_DIR", "")
BENCHMARK_FOLDER = os.getenv("TEST_BENCHMARK_DIR")
LOG_DIR = os.getenv("TEST_OUTPUT_DIR", "/mnt/output")
os.makedirs(LOG_DIR, exist_ok=True)

# Define constants for file names
RESULTS_FILE = "numerical_results.json"
WEIGHTS_FILE = "weights.safetensors"
BENCH_CONFIG_FILE = "bench_config.json"
TRACED_TENSORS_FILE = "traced_tensors.json"
COMPARE_RESULTS_FILE = "compare_results.json"

# Define constants for batch sizes, sequence lengths, and data types
BATCH_SIZES = [
    int(x) for x in os.getenv("TEST_BATCH_SIZES", "16,8,1").split(",") if x.strip()
]
SEQ_LENS = [
    int(x)
    for x in os.getenv("TEST_SEQ_LENS", "4096,2048,1024,512").split(",")
    if x.strip()
]
TEST_DTYPES = [torch.bfloat16]
# Define the target data type for comparison tests
TARGET_DTYPE = torch.bfloat16

RANDOM_INPUT_COUNT = int(os.getenv("TEST_RANDOM_INPUT_COUNT", "5"))
REPEAT_COUNT = int(os.getenv("TEST_REPEAT_COUNT", "3"))
