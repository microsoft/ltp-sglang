import argparse
import csv
from model_config import ModelConfig
from counter_registry import CounterRegistry


def main():
    parser = argparse.ArgumentParser(description="Model Info Tool")
    parser.add_argument("--config", type=str, default="../../model_conf/qwen/config.json,../../model_conf/deepseek/config.json", help="Path to the model configuration file")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length for the model")
    parser.add_argument("--counters", type=str, default="param", help="Comma-separated list of counters to use")
    args = parser.parse_args()
    
    seq_len = args.seq_len
    counter_names = args.counters.split(',')
    
    models_info = {}
    for config_path in args.config.split(','):
        config = ModelConfig.from_json_file(config_path)
        models_info[config.name] = {}
        for counter_name in counter_names:
            try:
                counter = CounterRegistry.get_counter(counter_name)
                models_info[config.name][counter.metric_name] = counter.get_model_results(seq_len, config)
            except (ValueError, IndexError) as e:
                print(f"Warning: {e}")
                continue
    with open("model_info.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Model"] + list(models_info[next(iter(models_info))].keys()))
        for model_name, metrics in models_info.items():
            writer.writerow([model_name] + list(metrics.values()))

if __name__ == "__main__":
    main()
            