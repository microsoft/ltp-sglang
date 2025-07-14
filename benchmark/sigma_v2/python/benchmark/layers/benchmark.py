import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0+PTX"  # Set CUDA architecture for PyTorch

import argparse
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from attn import AttentionBenchmark
from linear import LinearBenchmark
from exp import ExpBenchmark
from constants import layer_configs, layer_fig_groups, results_dir

available_benchmarks = {
    "attention": AttentionBenchmark,
    "linear": LinearBenchmark,
    "exp": ExpBenchmark,
}


def plot_comparison_figure(layer, df):
    """Generates and saves a comparison plot for benchmark results."""
    fig_groups = layer_fig_groups.get(layer, {})
    all_subplots = []
    mp_values = sorted(df['mp'].unique())

    for mp in mp_values:
        df_mp = df[df['mp'] == mp]
        for group_name, funcs in fig_groups.items():
            group_df = df_mp[df_mp['func'].isin(funcs)]
            if group_df.empty:
                continue

            # Pivot table with bsz_seq as rows and functions as columns
            pivot = group_df.pivot_table(index="bsz_seq", columns="func", values="dur_ms").sort_index()
            # Normalize each row by its fastest time
            normed = pivot.apply(lambda row: row / row.min(), axis=1)
            x_labels = list(normed.index.astype(str))
            all_subplots.append({
                "title": f"{layer} - MP {mp} - {group_name}",
                "normed": normed,
                "x_labels": x_labels,
                "n_funcs": len(funcs),
                "funcs": funcs
            })

    if not all_subplots:
        return

    n_plots = len(all_subplots)
    fig, axes = plt.subplots(n_plots, 1, figsize=(20, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for ax, subplot in zip(axes, all_subplots):
        normed = subplot["normed"]
        x_labels = subplot["x_labels"]

        # Sort x_labels based on (batch size, sequence length)
        def sort_key(label):
            bsz, seq = [int(x.strip()) for x in label.strip("()").split(',')]
            return (bsz, seq)

        x_labels_sorted = sorted(x_labels, key=sort_key)
        normed.index = normed.index.astype(str)
        normed = normed.reindex(x_labels_sorted)
        
        x = np.arange(len(x_labels_sorted))
        width = 0.8 / subplot["n_funcs"]
        max_y = 0.0

        for i, func in enumerate(subplot["funcs"]):
            if func in normed.columns:
                values = normed[func].fillna(0).values
                values[np.isinf(values)] = 0
                max_y = max(max_y, values.max())
                ax.bar(x + i * width, values, width, label=func)

        ax.axhline(1.0, color='red', linestyle='--')
        ax.set_xlim(x[0] - width * subplot["n_funcs"], x[-1] + width * subplot["n_funcs"])
        ax.set_ylim(0.8, max_y + 0.2)
        ax.set_ylabel("Normalized Duration")
        ax.set_xlabel("Bsz, SeqLen")
        ax.set_title(subplot["title"])
        ax.set_xticks(x + width * (subplot["n_funcs"] - 1) / 2)
        ax.set_xticklabels(x_labels_sorted, rotation=90, ha="right")
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{results_dir}/{layer}_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def save_results_to_csv(layer_results):
    """Saves benchmark results to CSV and generates comparison figures."""
    for layer, mp_results in layer_results.items():
        csv_path = f"{results_dir}/{layer}_benchmark_results.csv"
        records = []

        for mp_size, config_results in mp_results.items():
            for config_key, func_metrics in config_results.items():
                for func_name, metrics in func_metrics.items():
                    records.append({
                        "mp": mp_size,
                        "bsz_seq": config_key,
                        "func": func_name,
                        "dur_ms": metrics.get("time_ms"),
                        "gbps": metrics.get("bandwidth_gb_s"),
                        "tflops": metrics.get("throughput_tflops")
                    })

        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False, sep="\t")
        plot_comparison_figure(layer, df)


def load_existing_results(layers: list):
    """Loads existing results from CSV files and plots them."""
    for layer in layers:
        csv_path = f"{results_dir}/{layer}_benchmark_results.csv"
        try:
            df = pd.read_csv(csv_path, sep="\t")
            if df.empty:
                print(f"No existing results found for {layer}.")
                continue
            plot_comparison_figure(layer, df)
        except FileNotFoundError:
            print(f"No existing results found for {layer}. Please run the benchmark first.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark various layers")
    parser.add_argument("--layer", type=str, choices=["all", "attention", "linear", "exp"],
                        default="all", help="Type of layer to benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1],
                        help="Batch sizes to benchmark")
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                        default=[2048],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--mp-sizes", type=int, nargs="+", default=[8, 16],
                        help="Tensor/Expert parallel sizes")
    parser.add_argument("-f", "--force-rewrite", action="store_true",
                        help="Force rewrite of existing benchmark results")
    args = parser.parse_args()

    bszs = args.batch_sizes
    seq_lens = args.seq_lengths
    mp_sizes = args.mp_sizes

    layers = available_benchmarks.keys() if args.layer == "all" else [args.layer]

    if not args.force_rewrite:
        load_existing_results(layers)
    else:
        os.makedirs(results_dir, exist_ok=True)
        layer_results = {}

        for layer in layers:
            if layer not in available_benchmarks or layer not in layer_configs:
                print(f"Benchmark for {layer} is not available.")
                continue

            json_config_path = layer_configs.get(layer)
            with open(json_config_path, 'r') as f:
                config = json.load(f)

            benchmark_summary = {}
            for mp_size in mp_sizes:
                if mp_size == 1 and layer == "exp":
                    continue
                try:
                    benchmark = available_benchmarks[layer](mp_size=mp_size, **config)
                    results = benchmark.run_benchmarks(bszs, seq_lens)
                    if mp_size not in benchmark_summary:
                        benchmark_summary[mp_size] = {}
                    for config_key, metrics in results.items():
                        benchmark_summary[mp_size].setdefault(config_key, {}).update(metrics)
                except Exception as e:
                    print(f"Error benchmarking layer {layer} with mp_size {mp_size}: {e}")
                finally:
                    torch.cuda.empty_cache()

            layer_results[layer] = benchmark_summary

        save_results_to_csv(layer_results)


if __name__ == "__main__":
    main()