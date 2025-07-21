import json
from util import get_config_value

def get_net(model_config, seq_len=1, n_gpus=8):
    model_netio = {}
    for model, config_path in model_config.items():
        with open(config_path, "r") as f:
            config = json.load(f)
        total_layers = get_config_value(config, ["num_hidden_layers", "n_layer"], 0)
        hidden_size = get_config_value(config, ["hidden_size", "n_embd"], 768)
        vocab_size = get_config_value(config, ["vocab_size", "n_vocab"], 0)
        model_netio[model] = total_layers * seq_len * hidden_size * 2 + seq_len * vocab_size * 2 // n_gpus 
    return model_netio