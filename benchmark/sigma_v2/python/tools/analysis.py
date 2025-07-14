import pandas as pd
from constant import LOG_DIR, model_hardware, CSV_DIR
from flop import get_model_flops
from fig import plot_per_model_mfu
import os
import glob

def extract_latency(log_file):
    """
    Extract latency information from the log file.
    """
    prefill_latencies = []
    decode_latencies = []
    with open(log_file, "r") as file:
        for line in file:
            if "Using model: model_conf/" in line:
                # Using model: model_conf/config_fp16.json
                model_arch = line.split("model_conf/")[1].split(".json")[0]
            if "--model=./model_conf/" in line:
                tp = int(line.split("--tensor-parallel-size=")[1].split("\'")[0])
                dp = tp if "--enable-dp-attention" in line else 1
                ep = tp if "--enable-ep-moe" in line else 1
            if "Phoenix Device" in line:
                phase, bsz, seq, latency = line.split("Phoenix Device ")[1].split(" ")[1:5]
                bsz, seq, latency = int(bsz), int(seq), float(latency)
                bsz = bsz * dp
                model_name = model_arch.split("/")[0]
                model_name = "dpsk" if model_name == "deepseek" else model_name
                config_path = f"../../model_conf/{model_arch}.json"
                if phase == "Prefill":
                    flops = get_model_flops(model_name, config_path, bsz=bsz, seq_len=seq, kv_cache=0) / 1e12
                    mfu = flops / (float(latency) / 1000) / (model_hardware[model_name] * tp) * 100
                    prefill_latencies.append((tp, dp, ep, bsz, seq, latency, mfu)) 
                else:
                    flops = get_model_flops(model_name, config_path, bsz=bsz, seq_len=1, kv_cache=seq) / 1e12
                    mfu = flops / (float(latency) / 1000) / (model_hardware[model_name] * tp) * 100 
                    seq -= 1
                    decode_latencies.append((tp, dp, ep, bsz, seq, latency, mfu))          
    return prefill_latencies, decode_latencies

def main():
    model_log = {}
    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    for log_file in log_files:
        if 'benchmark.log' in log_file:
            continue
        filename = os.path.basename(log_file)
        fields = filename.split(".")[0].split("_")
        if 'qwen_sigma' in filename:
            model = fields[0]+'_' + fields[1]
            conf = fields[-1] if len(fields) > 2 else "tp8"
        else:
            model = fields[0]
            conf = fields[-1] if len(fields) > 1 else "tp8"
        if model not in model_log:
            model_log[model] = []
        model_log[model].append((conf, log_file))
    for model, logs in model_log.items():
        model_prefill_latencies = []
        model_decode_latencies = []
        for conf, log_file in logs:
            print(f"Processing {model} {conf}")
            prefill_latencies, decode_latencies = extract_latency(log_file)
            plot_per_model_mfu(model, conf, prefill_latencies, decode_latencies)
            model_prefill_latencies.extend(prefill_latencies)
            model_decode_latencies.extend(decode_latencies)
        prefill_df = pd.DataFrame(model_prefill_latencies, 
                                    columns=["tp", "dp", "ep", "bsz", "seq", "prefill latency", "prefill mfu"])
        decode_df = pd.DataFrame(model_decode_latencies,
                                    columns=["tp", "dp", "ep", "bsz", "seq", "decode latency", "decode mfu"])
        merged_df = pd.merge(prefill_df, decode_df, on=["tp", "dp", "ep", "bsz", "seq"], how="inner")
        merged_df.to_csv(os.path.join(CSV_DIR, f"{model}_latency.csv"), index=False)

if __name__ == "__main__":
    main()   