import yaml
from paths import BASE_DEPLOY_CONF_FILE

def retrieve_profile_settings() -> tuple:
    """
    Get the profile settings from the YAML config file.
    """
    with open(BASE_DEPLOY_CONF_FILE, "r") as f:
        config = yaml.safe_load(f)
    profile = config["enable_profile"]
    profile_phase = True if config["profile_phase"] == "prefill" else False
    return profile, profile_phase
