import json
import os

import torch

# Set the distributed environment variables
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

# Set the checkpoint path and log directory
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "path/to/checkpoint")
LOG_DIR = os.getenv("LOG_DIR", "comparison_test")
os.makedirs(LOG_DIR, exist_ok=True)

# Define constants for file names
WEIGHTS_FILE = "weights.safetensors"
TEST_CONFIG_FILE = "test_config.json"

# batch_size = [32, 8, 1]
# seq_len = [4096, 2048, 1024, 512, 128, 32, 8, 1]
BATCH_SIZES = [1]
SEQ_LENS = [4096, 2048, 1024, 512]
TEST_DTYPES = [torch.bfloat16]

RANDOM_INPUT_COUNT = 5
REPEAT_COUNT = 1


class ComparisonTestConfig:
    """Base class for test configurations."""

    module_config: dict = {}
    real_weight_prefix: str = ""
    log_dir: str = ""
    batch_sizes: list[int] = BATCH_SIZES
    seq_lens: list[int] = SEQ_LENS
    dtype: torch.dtype = torch.bfloat16
    input_count: int = RANDOM_INPUT_COUNT
    repeat_count: int = REPEAT_COUNT

    def __init__(
        self,
        module_config,
        real_weight_prefix="",
        log_dir="",
        batch_sizes=BATCH_SIZES,
        seq_lens=BATCH_SIZES,
        dtype=torch.bfloat16,
        input_count=RANDOM_INPUT_COUNT,
        repeat_count=REPEAT_COUNT,
    ):
        """Initialize the test configuration."""
        self.module_config = module_config
        self.real_weight_prefix = real_weight_prefix
        self.log_dir = log_dir
        self.batch_sizes = batch_sizes
        self.seq_lens = seq_lens
        self.dtype = dtype
        self.input_count = input_count
        self.repeat_count = repeat_count

    def save_json(self, path):
        """Save the test configuration to a JSON file."""
        config = {
            "module_config": self.module_config,
            "real_weight_prefix": self.real_weight_prefix,
            "log_dir": self.log_dir,
            "batch_sizes": self.batch_sizes,
            "seq_lens": self.seq_lens,
            "dtype": str(self.dtype),
            "input_count": self.input_count,
            "repeat_count": self.repeat_count,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
