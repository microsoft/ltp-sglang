import json

SEQ_LEN = 1024
TP = 8

def get_net(model_config, bsz=1, seq_len=SEQ_LEN):
    model_netio = {}
    for model, config_path in model_config.items():
        with open(config_path, "r") as f:
            config = json.load(f)
        layers = config.get("num_hidden_layers", config.get("n_layer"))
        if model in ["qwen", "dpsk", "qwen_sigma"]:
            hidden_size = config.get("hidden_size", config.get("n_embd", 768))
            vocab_size = config.get("vocab_size", 50257)
            model_netio[model] = (layers * seq_len * hidden_size * 2 + seq_len * vocab_size * 2 // TP) * bsz
        else:
            model_netio[model] = -1
    return model_netio