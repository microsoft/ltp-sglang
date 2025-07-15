from flop import get_flops
from params import get_params, get_param_size
from net import get_net
from mem import get_mem, get_model_mem_by_type
from constant import model_config, model_hardware, LATENCY_FILE, model_length
import argparse
import pandas as pd

def get_model_results(bsz, seq_len, model_latency={}):
    params = get_params(model_config)
    param_sizes = get_param_size(model_config)
    prefill_flops = get_flops(model_config, seq_len=seq_len, kv_cache=0)
    decode_flops = get_flops(model_config, kv_cache=seq_len)
    prefill_memio = get_mem(model_config, seq_len=seq_len, kv_cache=0)
    decode_memio = get_mem(model_config, kv_cache=seq_len)
    prefill_netio = get_net(model_config, seq_len=seq_len)
    decode_netio = get_net(model_config)
    rows = []
    for model in model_config.keys():
        row = {
            ('Param', 'B'): params[model] / 1e9,
            ('Param', 'GB'): param_sizes[model] / 1e9,
            ('Prefill', 'TFlops'): prefill_flops[model] / 1e12,
            ('Prefill', 'MemTB'): prefill_memio[model] / 1e12 ,
            ('Prefill', 'NetMB'): prefill_netio[model] / 1e6,
            ('Decode', 'TFlops'): decode_flops[model] / 1e12,
            ('Decode', 'MemTB'): decode_memio[model] / 1e12,
            ('Decode', 'NetMB'): decode_netio[model] / 1e6,
        }
        row["Model"] = model
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.set_index(("Model"))
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    print(df.round(4))
    df.to_csv('model_results.csv', index=True)
    return df

def get_model_param_cache(bsz=1, seq_len=-1):
    model_param_kv_cache = {}
    use_max = seq_len == -1
    for model in model_config.keys():
        max_length = model_length.get(model, 1024)
        model_seq_len = max_length if use_max else seq_len
        config_path = model_config[model]
        mem_by_type = get_model_mem_by_type(model, config_path, bsz=bsz, seq_len=1, kv_cache=model_seq_len)
        model_param_kv_cache["{}({})".format(model, model_seq_len)] = (mem_by_type["param"] / 1e9, mem_by_type["kv_cache"] / 1e9) 
    return model_param_kv_cache

def main():
    argparser = argparse.ArgumentParser(description="Model Performance Analysis")
    argparser.add_argument("--bsz", type=int, default=1, help="Batch size for the model")
    argparser.add_argument("--seq_len", type=int, default=1024, help="Sequence length for the model")
    args = argparser.parse_args()
    bsz = args.bsz
    seq_len = args.seq_len
    summary_df = get_model_results(bsz, seq_len)
    
if __name__ == "__main__":
    main()