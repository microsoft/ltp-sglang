import matplotlib.pyplot as plt
from constant import FIGURE_DIR
plt.rcParams.update({'font.size': 20})

def plot_per_model_mfu(model, conf, prefill_latencies, decode_latencies):
    fig, axes = plt.subplots(2, 1, figsize=(40, 5 * 2))
    plt.subplots_adjust(hspace=1.3)
    prefill_bszseq_mfu = {(v[3], v[4]): v[6] for v in prefill_latencies}
    decode_bszseq_mfu = {(v[3], v[4]): v[6] for v in decode_latencies}
    prefill_bszseq_mfu = dict(sorted(prefill_bszseq_mfu.items(), key=lambda item: (item[0][1], item[0][0])))
    decode_bszseq_mfu = dict(sorted(decode_bszseq_mfu.items(), key=lambda item: (item[0][1], item[0][0])))
    TP, DP, EP = prefill_latencies[0][0], prefill_latencies[0][1], prefill_latencies[0][2]
    axes[0].set_title(f"Prefill MFU")
    axes[1].set_title(f"Decode MFU")
    axes[0].set_xlabel("Global Batch Size, Seq Length")
    axes[1].set_xlabel("Global Batch Size, Cache Length")
    axes[0].plot(prefill_bszseq_mfu.values(), marker='o', markersize=10, label=f"{model} {conf} Prefill MFU")
    axes[1].plot(decode_bszseq_mfu.values(), marker='o', markersize=10, label=f"{model} {conf} Decode MFU")
    axes[0].set_ylabel("MFU (%)")
    axes[1].set_ylabel("MFU (%)")
    axes[0].set_xticks(range(len(prefill_bszseq_mfu)))
    axes[0].set_xticklabels([f"{bsz},{seq}" for bsz, seq in prefill_bszseq_mfu.keys()], rotation=90)
    axes[1].set_xticks(range(len(decode_bszseq_mfu)))
    axes[1].set_xticklabels([f"{bsz},{seq}" for bsz, seq in decode_bszseq_mfu.keys()], rotation=90)
    axes[0].grid()
    axes[1].grid()
    fig.savefig(f"{FIGURE_DIR}/{model}_TP{TP}_DP{DP}_EP{EP}_mfu.pdf", bbox_inches='tight')
    
def plot_per_model_latency(model, conf, prefill_latencies, decode_latencies):
    fig, axes = plt.subplots(2, 1, figsize=(40, 5 * 2))
    plt.subplots_adjust(hspace=1.3)
    prefill_bszseq_latency = {(v[3], v[4]): v[5] for v in prefill_latencies}
    decode_bszseq_latency = {(v[3], v[4]): v[5] for v in decode_latencies}
    prefill_bszseq_latency = dict(sorted(prefill_bszseq_latency.items(), key=lambda item: (item[0][1], item[0][0])))
    decode_bszseq_latency = dict(sorted(decode_bszseq_latency.items(), key=lambda item: (item[0][1], item[0][0])))
    TP, DP, EP = prefill_latencies[0][0], prefill_latencies[0][1], prefill_latencies[0][2]
    axes[0].set_title(f"TTFT")
    axes[1].set_title(f"TPOT")
    axes[0].set_xlabel("Global Batch Size, Seq Length")
    axes[1].set_xlabel("Global Batch Size, Cache Length")
    axes[0].plot(prefill_bszseq_latency.values(), marker='o', markersize=10, label=f"{model} {conf}")
    axes[1].plot(decode_bszseq_latency.values(), marker='o', markersize=10, label=f"{model} {conf}")
    axes[0].set_yscale('log')
    # axes[1].set_yscale('log')
    axes[0].set_ylabel("Time (ms)")
    axes[1].set_ylabel("Time (ms)")
    axes[0].set_xticks(range(len(prefill_bszseq_latency)))
    axes[0].set_xticklabels([f"{bsz},{seq}" for bsz, seq in prefill_bszseq_latency.keys()], rotation=90)
    axes[1].set_xticks(range(len(decode_bszseq_latency)))
    axes[1].set_xticklabels([f"{bsz},{seq}" for bsz, seq in decode_bszseq_latency.keys()], rotation=90)
    axes[0].grid()
    axes[1].grid()
    fig.savefig(f"{FIGURE_DIR}/{model}_TP{TP}_DP{DP}_EP{EP}_latency.pdf", bbox_inches='tight')
