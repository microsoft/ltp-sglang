import argparse
from model_config import ModelConfig
from counter_registry import CounterRegistry
import pandas as pd

def convert_to_df(models_info):
    df = pd.DataFrame.from_dict(models_info, orient='index')
    df.index.name = 'Model'
    df.reset_index(inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Model Info Tool")
    parser.add_argument("--config", type=str, default="../../model_conf/qwen_sigma/config.json", help="Path to the model configuration file")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length for the model")
<<<<<<< HEAD
    parser.add_argument("--counters", type=str, default="param,prefill_mem,decode_mem", help="Comma-separated list of counters to use")
=======
    parser.add_argument("--counters", type=str, default="param", help="Comma-separated list of counters to use")
    parser.add_argument("-s", "--save", action="store_true", help="Save the model information to a CSV file")
    parser.add_argument("-o", "--output", type=str, default="model_info.csv", help="Output file name for saving model information")
>>>>>>> 3e0c62ce39c5363708a5ae16cafa711cb701bc46
    args = parser.parse_args()
    
    seq_len = args.seq_len
    counter_names = args.counters.split(',')
    models_info = {}
    for config_path in args.config.split(','):
        print(f"Processing config: {config_path}")
        config = ModelConfig.from_json_file(config_path)
        models_info[config.name] = {}
        for counter_name in counter_names:
            try:
<<<<<<< HEAD
                if 'mem' in counter_name:
                    counter = CounterRegistry.get_counter(counter_name, precision_bytes=config.precision_bytes, quantization_block_size=config.quantization_block_size)
                else:
                    counter = CounterRegistry.get_counter(counter_name)
                models_info[config.name][counter.metric_name] = counter.get_model_results(seq_len, config)
=======
                counter = CounterRegistry.get_counter(counter_name)
                result = counter.get_model_results(seq_len, config)
                models_info[config.name][counter.metric_name] = result
>>>>>>> 3e0c62ce39c5363708a5ae16cafa711cb701bc46
            except (ValueError, IndexError) as e:
                print(f"    Warning: {e}")
                models_info[config.name][counter.metric_name] = -1
                continue
    df = convert_to_df(models_info)
    print(df)
    if args.save:
        print(f"Saving model information to: {args.output}")
        df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()