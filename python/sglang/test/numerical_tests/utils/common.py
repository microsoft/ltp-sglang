import os

import torch

# Set the distributed environment variables
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

# Set the checkpoint path and log directory
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "path/to/checkpoint")
LOG_DIR = os.getenv("LOG_DIR", "numerical_test")
RESULTS_FILE = "numerical_results.json"
os.makedirs(LOG_DIR, exist_ok=True)

BATCH_SIZES = [16, 8, 1]
SEQ_LENS = [4096, 2048, 1024, 512]
TEST_DTYPES = [torch.bfloat16]

RANDOM_INPUT_COUNT = 1
REPEAT_COUNT = 3
