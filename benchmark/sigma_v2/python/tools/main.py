from flop import get_flops
from params import get_params, get_param_size
from net import get_net
from mem import get_mem, get_model_mem_by_type
from constant import model_config, model_hardware, LATENCY_FILE, model_length
import argparse
import pandas as pd


def get_model_latency(bsz, seq_len):
    with open(LATENCY_FILE, 'r') as f:
        lines = f.readlines()
    model_latency = {}
    for l in lines:
        fields = l.split()
        model, model_bsz, model_seq_len, prefill_latency, decode_latency = fields[0], int(fields[1]), int(fields[2]), float(fields[3]), float(fields[4])
        if bsz == model_bsz and seq_len == model_seq_len:
            model_latency[model] = (prefill_latency, decode_latency)
    return model_latency

def get_model_results(bsz, seq_len, model_latency):
    params = get_params(model_config)
    param_sizes = get_param_size(model_config)
    prefill_flops = get_flops(model_config, bsz=bsz, seq_len=seq_len, kv_cache=0)
    decode_flops = get_flops(model_config, bsz=bsz, seq_len=1, kv_cache=seq_len)
    prefill_memio = get_mem(model_config, bsz=bsz, seq_len=seq_len, kv_cache=0)
    decode_memio = get_mem(model_config, bsz=bsz,seq_len=1, kv_cache=seq_len)
    prefill_netio = get_net(model_config, bsz=bsz, seq_len=seq_len)
    decode_netio = get_net(model_config, bsz=bsz, seq_len=1)
    rows = []
    for model in model_config.keys():
        b_params = params[model] / 1e9  # Convert to billions
        gb_params = param_sizes[model] / 1e9  # Convert to GB
        b_prefill_flops = prefill_flops[model] / 1e12  # Convert to TFlops
        b_decode_flops = decode_flops[model] / 1e12  # Convert to TFlops
        prefill_flops_per_param = b_prefill_flops / b_params if b_params > 0 else 0
        decode_flops_per_param = b_decode_flops / b_params if b_params > 0 else 0
        gb_prefill_mem = prefill_memio[model] / 1e12 # Convert to GB
        gb_decode_mem = decode_memio[model] / 1e12 # Convert to GB
        prefill_byte_per_flops = gb_prefill_mem / b_prefill_flops if b_prefill_flops > 0 else 0
        decode_byte_per_flops = gb_decode_mem / b_decode_flops if b_decode_flops > 0 else 0
        gb_prefill_net = prefill_netio[model] / 1e9  # Convert to GB
        gb_decode_net = decode_netio[model] / 1e9  # Convert to GB
        prefill_flops_per_comm_byte = b_prefill_flops / gb_prefill_net if gb_prefill_net > 0 else 0
        decode_flops_per_comm_byte = b_decode_flops / gb_decode_net if gb_decode_net > 0 else 0
        prefill_latency, decode_latency = model_latency.get(model, (0, 0))
        hardware_flops = model_hardware.get(model, 989.5)  # Default to 989.5 TFlops if not specified
        prefill_mfu = b_prefill_flops / (prefill_latency / 1000) / (hardware_flops * 8) * 100 if prefill_latency > 0 else 0
        decode_mfu = b_decode_flops / (decode_latency / 1000) / (hardware_flops * 8) * 100 if decode_latency > 0 else 0
        row = {
            ('Param', 'B'): b_params,
            ('Param', 'GB'): gb_params,
            ('Prefill', 'TFlops'): b_prefill_flops,
            ('Prefill', 'TF/Param'): prefill_flops_per_param,
            ('Prefill', 'MFU (%)'): prefill_mfu,
            ('Prefill', 'MemTB'): gb_prefill_mem,
            ('Prefill', 'MemTB/TF'): prefill_byte_per_flops,
            ('Prefill', 'NetMB'): gb_prefill_net * 1000,
            ('Prefill', 'TF/NetGB'): prefill_flops_per_comm_byte,
            ('Decode', 'TFlops'): b_decode_flops,
            ('Decode', 'TF/Param'): decode_flops_per_param,
            ('Decode', 'MFU (%)'): decode_mfu,
            ('Decode', 'MemTB'): gb_decode_mem,
            ('Decode', 'MemTB/TF'): decode_byte_per_flops,
            ('Decode', 'NetMB'): gb_decode_net * 1000,
            ('Decode', 'TF/NetGB'): decode_flops_per_comm_byte,
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
    plot_hardware_flops_per_gb()
    plot_hardware_flops_per_membw()
    plot_hardware_flops_per_nvlink_bw()
    bsz = args.bsz
    seq_len = args.seq_len
    model_latency = get_model_latency(bsz, seq_len)
    summary_df = get_model_results(bsz, seq_len, model_latency)
    
if __name__ == "__main__":
    main()