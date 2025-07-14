from pathlib import Path

current_file = Path(__file__).resolve()
BASE_DIR = current_file.parents[2]
LOG_FILE = BASE_DIR / "log" / "benchmark.log"

DEPLOY_CONF_DIR = BASE_DIR / "conf" / "run"
BASE_DEPLOY_CONF_FILE = BASE_DIR / "conf" / "default.yaml"