import yaml
from pathlib import Path
current_file = Path(__file__).resolve()
BASE_DIR = current_file.parents[2]
BASE_CONF_FILE = BASE_DIR / "conf" / "default.yaml"

def retrieve_profile_settings() -> tuple:
    """
    Get the profile settings from the YAML config file.
    """
    with open(BASE_CONF_FILE, "r") as f:
        config = yaml.safe_load(f)
    profile = config["enable_profile"]
    profile_phase = True if config["profile_phase"] == "prefill" else False
    return profile, profile_phase
