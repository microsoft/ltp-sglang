import sys
latency_results = './results/csv/latency_benchmark.csv'
def main():
    log_file = sys.argv[1]
    with open(log_file, 'r') as f:
        lines = f.readlines()
    model = None
    model_config = None
    tp, ep, dp = 1, 1, 1
    attn = "fa3"
    csv_lines = []
    for line in lines:
        if 'Using model' in line:
            fields = line.split(': ')[1].strip().split('/')
            model, model_config = fields[1], fields[2].split('.')[0].split('_')[1]
        if "attention-backend" in line:
            #['--model=./model_conf/qwen_sigma', '--log-level=error', '--load-format=dummy', '--chunked-prefill-size=1048576', '--max-prefill-tokens=1048576', '--attention-backend=fa3', '--mem-fraction-static=0.5', '--max-running-requests=16384', '--dist-init-addr=10.0.0.101:30000', '--nnodes=1', '--node-rank=0', '--tensor-parallel-size=8', '--enable-ep-moe', '--ep-size=8', '--trust-remote-code', '--disable-radix-cache', '--disable-overlap-schedule']
            attn = line.split('attention-backend=')[1].split('\'')[0]
            tp = int(line.split('tensor-parallel-size=')[1].split('\'')[0])
            if "ep-size" in line:
                ep = int(line.split('ep-size=')[1].split('\'')[0])
            if "data-parallel-size" in line:
                dp = int(line.split('data-parallel-size=')[1].split('\'')[0])
        if 'Latency Benchmark Device 0' in line:
            phase, bsz, seq_len, latency = line.split()[4:8]
            csv_lines.append(
                f"{model},{model_config},{attn},{tp},{ep},{dp},{phase},{bsz},{seq_len},{latency}\n"
            )
    with open(latency_results, 'a') as f:
        f.writelines(csv_lines)
if __name__ == "__main__":
    main()