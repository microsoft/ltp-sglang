import os
import re
import argparse
from model_config import ModelConfig
from counter_registry import CounterRegistry

gpu_flops_map = {
    "h200": 989.5,
}

flop_counter = {
    "Prefill": CounterRegistry.get_counter("prefill_flop"),
    "Decode": CounterRegistry.get_counter("decode_flop")
}

def calculate_mfu(model_config, deploy, phase, bsz, seq_len, latency, gpu_flops):
    gpus = int(re.search(r"tp(\d+)", deploy).group(1))
    flops = flop_counter[phase].get_model_results(seq_len, model_config) * bsz
    mfu = flops / (gpu_flops * gpus) / (latency / 1e3) if latency != 0 else 0
    return mfu

def analyse_results(log_file, gpu_flops):
    with open(log_file, 'r') as file:
        lines = file.readlines()
    results = []
    model, arch, base, deploy, backend = None, None, None, None, None
    model_config = None
    for l in lines: 
        if "Using model config" in l:
            fields = l.split("Using model config: ")[1].split(" ")[0:2]
            model, arch = fields[0], fields[1].strip()
            config_path = f"model_conf/{model}/config_{arch}.json"
            model_config = ModelConfig.from_json_file(config_path)
        if "Using configs" in l:
            fields = l.split("Using configs: ")[1].strip().split(", ")
            base, deploy, backend = fields[0], fields[1], fields[2]
        if "Latency Benchmark Device 0" in l:
            fields = l.split("Latency Benchmark")[1].split()
            phase, bsz, seq_len, latency = fields[3].rstrip(','), int(fields[6].rstrip(',')), int(fields[9].rstrip(',')), float(fields[12])
            mfu = calculate_mfu(model_config, deploy, phase, bsz, seq_len, latency, gpu_flops)
            results.append("{},{},{},{},{},{},{},{},{},{}".format(model, arch, base, deploy, backend, phase, bsz, seq_len, latency, mfu))
    return results

def main():
    parser = argparse.ArgumentParser(description="Run benchmark with specified configurations")
    parser.add_argument("--gpu", type=str, choices=["h200"], help="GPU type")
    parser.add_argument("--log", type=str, default="results/raw_output/benchmark.log", help="Log file path")
    parser.add_argument("--csv", type=str, default="results/csv/benchmark.csv", help="CSV file path")
    args = parser.parse_args()
    if not os.path.exists(args.log):
        print(f"Log file {args.log} does not exist.")
        return
    gpu_flops = gpu_flops_map[args.gpu]
    results = analyse_results(args.log, gpu_flops)
    if results:
        if not os.path.exists(os.path.dirname(args.csv)):
            os.makedirs(os.path.dirname(args.csv))
        file_exists = os.path.exists(args.csv)
        with open(args.csv, 'a' if file_exists else 'w') as f:
            for result in results:
                f.write(result + '\n')
        print(f"Results written to {args.csv}")
        
if __name__ == "__main__":
    main()