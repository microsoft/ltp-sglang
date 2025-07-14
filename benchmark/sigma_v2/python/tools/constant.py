LATENCY_FILE = './latency'
LOG_DIR = '../../log/full'
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

hardware_flops_mem_cap = {
    "V100": 7.8 / 32,  # TFlops per GB
    "A100": 312 / 80,  # TFlops per GB
    "H200": 989 / 141,  # TFlops per GB
    "GB300": 5000 / 597,  # TFlops per GB
}

hardware_flops_mem_bw = {
    "V100": 1/ 7.8 / 900,  # TFlops per GB
    "A100": 1/312 / 2039,  # TFlops per GB
    "H200": 1/989 / 4800,  # TFlops per GB
    "GB300": 1/5000 / 16000,  # TFlops per GB
}

hardware_flops_nvlink = {
    "V100": 7.8 / 300,  # TFlops per GB
    "A100": 312 / 600,  # TFlops per GB
    "H200": 989 / 900,  # TFlops per GB
    "GB300": 5000 / 3600,  # TFlops per GB
}