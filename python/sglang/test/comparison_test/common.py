import json
import os

import torch

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "path/to/checkpoint")

LOG_DIR = os.getenv("LOG_DIR", "comparison_test")
os.makedirs(LOG_DIR, exist_ok=True)

# Define constants for file names
WEIGHTS_FILE = "weights.safetensors"
TEST_CONFIG_FILE = "test_config.json"

# batch_size = [32, 8, 1]
# seq_len = [4096, 2048, 1024, 512, 128, 32, 8, 1]
BATCH_SIZES = [1]
SEQ_LENS = [8]
TEST_DTYPES = [torch.bfloat16]

RANDOM_INPUT_COUNT = 4
REPEAT_COUNT = 1


class TestConfig:
    """Base class for test configurations."""

    module_config = {}
    real_weight_prefix = ""
    log_dir = ""
    batch_sizes = BATCH_SIZES
    seq_lens = SEQ_LENS
    dtype = torch.bfloat16
    input_count = RANDOM_INPUT_COUNT
    repeat_count = REPEAT_COUNT

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
