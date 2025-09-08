# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse

import pandas as pd
from counter_registry import CounterRegistry
from model_config import ModelConfig


def convert_to_df(models_info):
    df = pd.DataFrame.from_dict(models_info, orient="index")
    df.index.name = "Model"
    df.reset_index(inplace=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Model Info Tool")
    parser.add_argument(
        "--config",
        type=str,
        default="../../model_conf/qwen_sigma/config.json",
        help="Path to the model configuration file",
    )
    parser.add_argument(
        "--seq_len", type=int, default=1024, help="Sequence length for the model"
    )
    parser.add_argument(
        "--counters",
        type=str,
        default="param",
        help="Comma-separated list of counters to use",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save the model information to a CSV file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="model_info.csv",
        help="Output file name for saving model information",
    )
    args = parser.parse_args()

    seq_len = args.seq_len
    counter_names = args.counters.split(",")
    models_info = {}
    for config_path in args.config.split(","):
        print(f"Processing config: {config_path}")
        config = ModelConfig.from_json_file(config_path)
        models_info[config.name] = {}
        for counter_name in counter_names:
            try:
                if "mem" in counter_name:
                    counter = CounterRegistry.get_counter(
                        counter_name, precision_bytes=config.precision_bytes
                    )
                else:
                    counter = CounterRegistry.get_counter(counter_name)
                result = counter.get_model_results(seq_len, config)
                models_info[config.name][counter.metric_name] = result
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
