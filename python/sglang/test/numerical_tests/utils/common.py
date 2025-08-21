import json
import os

import torch

# Set the distributed environment variables
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

# Set the checkpoint path and log directory
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
BENCHMARK_FOLDER = os.getenv("BENCHMARK_FOLDER")
LOG_DIR = os.getenv("LOG_DIR", "numerical_test")
os.makedirs(LOG_DIR, exist_ok=True)

# Define constants for file names
RESULTS_FILE = "numerical_results.json"
WEIGHTS_FILE = "weights.safetensors"
BENCH_CONFIG_FILE = "bench_config.json"
TRACED_TENSORS_FILE = "traced_tensors.json"
COMPARE_RESULTS_FILE = "compare_results.json"

# Define constants for batch sizes, sequence lengths, and data types
BATCH_SIZES = [16, 8, 1]
SEQ_LENS = [4096, 2048, 1024, 512]
TEST_DTYPES = [torch.bfloat16]
# Define the target data type for comparison tests
TARGET_DTYPE = torch.bfloat16

RANDOM_INPUT_COUNT = 5
REPEAT_COUNT = 3
