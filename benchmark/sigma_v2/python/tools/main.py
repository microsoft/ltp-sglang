import argparse
import pandas as pd
from typing import Dict
from flop import get_flops
from params import get_params, get_param_size
from mem import get_mem


def get_model_results(model_config: Dict, seq_len: int = 1024) -> pd.DataFrame:
    params = get_params(model_config)
    param_sizes = get_param_size(model_config)
    prefill_flops = get_flops(model_config, seq_len=seq_len, kv_cache=0)
    decode_flops = get_flops(model_config, kv_cache=seq_len)
    prefill_memio = get_mem(model_config, seq_len=seq_len, kv_cache=0)
    decode_memio = get_mem(model_config, kv_cache=seq_len)
    rows = []
    for model in model_config.keys():
        row = {
            ('Param', 'B'): params[model] / 1e9,
            ('Param', 'GB'): param_sizes[model] / 1e9,
            ('Prefill', 'TFlops'): prefill_flops[model] / 1e12,
            ('Prefill', 'MemTB'): prefill_memio[model] / 1e12 ,
            ('Decode', 'TFlops'): decode_flops[model] / 1e12,
            ('Decode', 'MemTB'): decode_memio[model] / 1e12,
        }
        row["Model"] = model
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.set_index(("Model"))
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    print(df.round(4))
    return df

def main():
    argparser = argparse.ArgumentParser(description="Model  Analysis")
    argparser.add_argument("--model", type=str, default="dpsk,qwen_sigma", help="Model name to analyze")
    argparser.add_argument("--config", type=str, default="../../model_conf/deepseek/config.json,../../model_conf/qwen_sigma/config.json", help="Path to the model configuration file")
    argparser.add_argument("--seq_len", type=int, default=1024, help="Sequence length for the model")
    args = argparser.parse_args()
    seq_len = args.seq_len
    model_config = {}
    for model, config in zip(args.model.split(","), args.config.split(",")):
        model_config[model] = config
    summary_df = get_model_results(model_config, seq_len)
    summary_df.to_csv('model_results.csv', index=True)

if __name__ == "__main__":
    main()