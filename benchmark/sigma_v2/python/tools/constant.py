LATENCY_FILE = './latency'
LOG_DIR = '../../log/'
FIGURE_DIR = '../../results/figures'
CSV_DIR = '../../results/csv'
PRETTY_LABEL = ["Qwen3", "DeepSeek V3", "Qwen Sigma"]

model_config = {
    "qwen": "../../model_conf/qwen/config_full.json",
    "dpsk": "../../model_conf/deepseek/config_full.json",
    "qwen_sigma": "../../model_conf/qwen_sigma/config_full.json",
}

model_hardware = {
    "qwen": 989.5, 
    "dpsk": 989.5 * 2,
    "qwen_sigma": 989.5, 
}

model_precision = {
    "qwen": 2,
    "dpsk": 1,
    "qwen_sigma": 2, 
}

model_length = {
    "qwen": 131072,
    "dpsk": 163840,
    "qwen_sigma": 4096, 
}