import argparse
from model_config import ModelConfig
from util import get_model_results
import params as ParamCounter
import csv

model_counters = {
    "Param": ParamCounter,
}

def main():
    parser = argparse.ArgumentParser(description="Model Info Tool")
    parser.add_argument("--config", type=str, default="../../model_conf/qwen/config.json,../../model_conf/deepseek/config.json", help="Path to the model configuration file")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length for the model")
    args = parser.parse_args()
    seq_len = args.seq_len
    models_info = {}
    for config_path in args.config.split(','):
        name, config = ModelConfig.from_json_file(config_path)
        models_info[name] = {}
        for metrics, counter in model_counters.items():
            models_info[name][metrics] = get_model_results(counter, seq_len, config)
    with open("model_info.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Model"] + list(models_info[next(iter(models_info))].keys()))
        for model_name, metrics in models_info.items():
            writer.writerow([model_name] + list(metrics.values()))

if __name__ == "__main__":
    main()
            