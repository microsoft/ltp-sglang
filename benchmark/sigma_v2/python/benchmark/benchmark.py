# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import asyncio
import dataclasses
import json
import logging
import os
from typing import List

import numpy as np
from omegaconf import OmegaConf
import time
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def fetch_server_info(tokenizer_manager):
    """Safely try to get server_info from tokenizer_manager.

    Some TokenizerManager variants do not expose `get_server_info`. Return None
    when not available or on error.
    """
    try:
        if hasattr(tokenizer_manager, "get_server_info"):
            server_info = tokenizer_manager.get_server_info()
            if hasattr(server_info, "__await__"):
                server_info = asyncio.get_event_loop().run_until_complete(server_info)
            return server_info
    except Exception:
        return None
    return None


def launch_inference(
    args: List,
    bszs: List,
    seq_lens: List,
    vocab_size: int,
    max_new_tokens: List,
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
):
    """
    Launches SGLang inference subprocesses for benchmarking.

    Args:
        args (List): List of command-line arguments for SGLang.
        bszs (List): List of batch sizes to test.
        seq_lens (List): List of sequence lengths to test.
        vocab_size (int): Vocabulary size for input token generation.
        max_new_tokens (int): Maximum number of new tokens to generate per request.
        warmup_iters (int): Number of warmup iterations. Default is 3.
        benchmark_iters (int): Number of benchmark iterations. Default is 10.
    """
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.server_args import prepare_server_args

    server_args = prepare_server_args(args)
    # Enable metrics collection
    server_args.enable_metrics = True
    logger.info(server_args)

    # Calculate total number of GPUs for per-GPU throughput
    # Note: EP (expert parallel) is handled within TP, not as separate GPUs
    # DP (data parallel) behavior depends on enable_dp_attention:
    #   - Without DP attention: Each DP replica uses separate GPUs → total = TP × DP × PP
    #   - With DP attention: All DP replicas share same GPUs → total = TP × PP
    tp_size = server_args.tp_size if hasattr(server_args, 'tp_size') else 1
    pp_size = server_args.pp_size if hasattr(server_args, 'pp_size') else 1
    dp_size = server_args.dp_size if hasattr(server_args, 'dp_size') else 1
    ep_size = server_args.ep_size if hasattr(server_args, 'ep_size') else 1
    enable_dp_attention = server_args.enable_dp_attention if hasattr(server_args, 'enable_dp_attention') else False

    # Calculate total GPUs based on DP mode
    if enable_dp_attention:
        # DP attention: All DP replicas share the same TP×PP GPUs
        total_gpus = tp_size * pp_size
    else:
        # Standard DP: Each DP replica gets its own set of TP×PP GPUs
        total_gpus = tp_size * pp_size * dp_size

    try:
        backend = Engine(**dataclasses.asdict(server_args))
        tokenizer_manager = backend.tokenizer_manager
        metrics_collector = tokenizer_manager.metrics_collector

        async def runner():
            response = None
            async for item in tokenizer_manager.generate_request(request):
                response = item

            return response

        for batch_size in bszs:
            for seq in seq_lens:
                for max_new_token in max_new_tokens:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Configuration: batch_size={batch_size}, seq_len={seq}, output_len={max_new_token}")
                    logger.info(f"{'='*60}")

                    # Warmup phase
                    logger.info(f"\nWarming up with {warmup_iters} iterations...")
                    for warmup_iter in range(warmup_iters):
                        input_ids = [
                            [int(x) for x in np.random.randint(0, high=vocab_size, size=(seq,))]
                            for _ in range(batch_size)
                        ]
                        request = GenerateReqInput(
                            input_ids=input_ids,
                            sampling_params={
                                "temperature": 0,
                                "max_new_tokens": max_new_token,
                            },
                        )
                        asyncio.get_event_loop().run_until_complete(runner())
                        logger.info(f"  Warmup iteration {warmup_iter + 1}/{warmup_iters} completed")

                    # Clear metrics after warmup
                    if metrics_collector:
                        metrics_collector.clear()

                    # Benchmark phase with multiple iterations
                    logger.info(f"\nRunning {benchmark_iters} benchmark iterations...")
                    benchmark_latencies = []

                    for bench_iter in range(benchmark_iters):
                        input_ids = [
                            [int(x) for x in np.random.randint(0, high=vocab_size, size=(seq,))]
                            for _ in range(batch_size)
                        ]
                        request = GenerateReqInput(
                            input_ids=input_ids,
                            sampling_params={
                                "temperature": 0,
                                "max_new_tokens": max_new_token,
                            },
                        )

                        st = time.perf_counter()
                        ret = asyncio.get_event_loop().run_until_complete(runner())
                        latency = time.perf_counter() - st
                        benchmark_latencies.append(latency)
                        logger.info(f"  Benchmark iteration {bench_iter + 1}/{benchmark_iters}: {latency:.4f}s")

                    # Use average latency for throughput calculations
                    avg_latency = np.mean(benchmark_latencies)
                    latency = avg_latency

                    # Extract TTFT and e2e_latency from meta_info
                    try:
                        meta_info = ret[0]["meta_info"]
                        if "e2e_latency" in meta_info:
                            logger.info(f"Latency in Response (meta): {meta_info['e2e_latency']}")
                    except Exception as e:
                        logger.error(f"Failed to extract metrics from response: {e}")

                    try:
                        server_info = fetch_server_info(backend)
                        last_gen_throughput = server_info["internal_states"][0][
                            "last_gen_throughput"
                        ]
                        # Extract memory usage information
                        memory_info = server_info["internal_states"][0].get("memory_usage", {})
                        weight_mem = memory_info.get("weight", 0)
                        kvcache_mem = memory_info.get("kvcache", 0)
                        cuda_graph_mem = memory_info.get("cuda_graph", 0)
                        token_capacity = memory_info.get("token_capacity", 0)
                    except Exception as e:
                        logger.error(f"Failed to fetch server info: {e}")
                        last_gen_throughput = None
                        memory_info = {}
                        weight_mem = kvcache_mem = cuda_graph_mem = token_capacity = 0

                    # Get current GPU memory usage
                    try:
                        if torch.cuda.is_available():
                            gpu_id = 0  # Assuming first GPU
                            torch.cuda.empty_cache()
                            free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
                            used_mem = (total_mem - free_mem) / (1024**3)  # Convert to GB
                            avail_mem = free_mem / (1024**3)  # Convert to GB
                            total_mem_gb = total_mem / (1024**3)  # Convert to GB
                        else:
                            used_mem = avail_mem = total_mem_gb = 0
                    except Exception as e:
                        logger.error(f"Failed to get GPU memory info: {e}")
                        used_mem = avail_mem = total_mem_gb = 0

                    # Compute throughput-style summary metrics (total input/output tokens, throughputs)
                    try:
                        total_input_tokens = sum(len(ids) for ids in input_ids)
                    except Exception:
                        total_input_tokens = batch_size * seq

                    try:
                        total_output_tokens = sum(
                            int(item.get("meta_info", {}).get("completion_tokens", 0))
                            for item in ret
                        )
                    except Exception:
                        total_output_tokens = 0

                    # Calculate throughput metrics using actual timing data from metrics collector
                    request_throughput = None
                    input_throughput = None
                    output_throughput = None
                    total_throughput = None
                    prefill_throughput = None
                    decode_throughput = None
                    # Per-GPU throughput metrics
                    per_gpu_total_throughput = None
                    per_gpu_prefill_throughput = None
                    per_gpu_decode_throughput = None

                    try:
                        # Get average metrics from collector for this benchmark iteration
                        if metrics_collector:
                            ttft_list = metrics_collector.time_to_first_token_list
                            prefill_lat_list = metrics_collector.prefill_latency_list
                            decode_lat_list = metrics_collector.decode_latency_list

                            avg_ttft = np.mean(ttft_list) if ttft_list else None
                            # Latencies already in ms, convert to seconds for throughput calculation
                            avg_prefill = (np.mean(prefill_lat_list) / 1000.0) if prefill_lat_list else None
                            avg_decode = (np.mean(decode_lat_list) / 1000.0) if decode_lat_list else None
                        else:
                            avg_ttft = None
                            avg_prefill = None
                            avg_decode = None

                        if latency > 0:
                            # Request throughput: requests / total time
                            request_throughput = batch_size / float(latency)

                            # Total throughput: all tokens / total time (system-wide throughput across all GPUs)
                            total_throughput = (total_input_tokens + total_output_tokens) / float(latency)
                            if total_gpus > 1:
                                per_gpu_total_throughput = total_throughput / total_gpus

                            # Input (prefill) throughput: total input tokens / average prefill time
                            # This is the system-wide throughput (across all TP/EP GPUs)
                            # avg_prefill is measured on the entire system, not per GPU
                            if avg_prefill and avg_prefill > 0:
                                input_throughput = total_input_tokens / avg_prefill
                                prefill_throughput = input_throughput  # Same as input throughput
                                if total_gpus > 1:
                                    per_gpu_prefill_throughput = prefill_throughput / total_gpus

                            # Output (decode) throughput: total output tokens / average decode time
                            # This is the system-wide throughput (across all TP/EP GPUs)
                            # avg_decode is measured on the entire system, not per GPU
                            if avg_decode and avg_decode > 0:
                                output_throughput = total_output_tokens / avg_decode
                                decode_throughput = output_throughput  # Same as output throughput
                                if total_gpus > 1:
                                    per_gpu_decode_throughput = decode_throughput / total_gpus
                    except Exception as e:
                        logger.error(f"Failed to calculate throughput: {e}")

                    # Print summary similar to bench_offline_throughput
                    logger.info("\n{:{c}^{n}}".format(" Offline Benchmark Result ", c="=", n=50))
                    logger.info(f"Backend: {'engine'}")
                    logger.info(f"Successful requests: {batch_size}")
                    logger.info(f"Benchmark duration (s): {latency:.2f}")
                    logger.info(f"Total input tokens: {total_input_tokens}")
                    logger.info(f"Total generated tokens: {total_output_tokens}")
                    dp_mode = "DP-Attention" if enable_dp_attention else "Standard-DP"
                    logger.info(f"Configuration: TP={tp_size}, PP={pp_size}, EP={ep_size}, DP={dp_size} ({dp_mode}), Total GPUs={total_gpus}")
                    logger.info("")
                    logger.info("System-wide Throughput (across all GPUs):")
                    if request_throughput is not None:
                        logger.info(f"  Request throughput (req/s): {request_throughput:.2f}")
                    if total_throughput is not None:
                        logger.info(f"  Total token throughput (tok/s): {total_throughput:.2f}")
                    if input_throughput is not None:
                        logger.info(f"  Input token throughput (tok/s): {input_throughput:.2f}")
                    if output_throughput is not None:
                        logger.info(f"  Output token throughput (tok/s): {output_throughput:.2f}")
                    if prefill_throughput is not None:
                        logger.info(f"  Prefill throughput (tok/s): {prefill_throughput:.2f}")
                    if decode_throughput is not None:
                        logger.info(f"  Decode throughput (tok/s): {decode_throughput:.2f}")
                    if last_gen_throughput is not None:
                        logger.info(f"  Last generation throughput (tok/s): {last_gen_throughput:.2f}")

                    if total_gpus > 1:
                        logger.info("")
                        logger.info(f"Per-GPU Throughput (average per GPU, {total_gpus} GPUs total):")
                        if per_gpu_total_throughput is not None:
                            logger.info(f"  Total token throughput per GPU (tok/s): {per_gpu_total_throughput:.2f}")
                        if per_gpu_prefill_throughput is not None:
                            logger.info(f"  Prefill throughput per GPU (tok/s): {per_gpu_prefill_throughput:.2f}")
                        if per_gpu_decode_throughput is not None:
                            logger.info(f"  Decode throughput per GPU (tok/s): {per_gpu_decode_throughput:.2f}")

                    # Print memory information
                    logger.info("\n{:{c}^{n}}".format(" Memory Usage ", c="=", n=50))
                    if total_mem_gb > 0:
                        logger.info(f"GPU Total Memory: {total_mem_gb:.2f} GB")
                        logger.info(f"GPU Used Memory: {used_mem:.2f} GB ({used_mem/total_mem_gb*100:.1f}%)")
                        logger.info(f"GPU Available Memory: {avail_mem:.2f} GB ({avail_mem/total_mem_gb*100:.1f}%)")
                    if weight_mem > 0:
                        logger.info(f"Model Weight Memory: {weight_mem:.2f} GB")
                    if kvcache_mem > 0:
                        logger.info(f"KV Cache Memory: {kvcache_mem:.2f} GB")
                    if cuda_graph_mem > 0:
                        logger.info(f"CUDA Graph Memory: {cuda_graph_mem:.2f} GB")
                    if token_capacity > 0:
                        logger.info(f"Token Capacity: {token_capacity}")

                    # Extract metrics from collector
                    if metrics_collector:
                        ttft_list = metrics_collector.time_to_first_token_list
                        itl_list = metrics_collector.inter_token_latency_seconds_list
                        tpot_list = metrics_collector.tpot_list
                        prefill_lat_list = metrics_collector.prefill_latency_list
                        decode_lat_list = metrics_collector.decode_latency_list

                        if ttft_list:
                            ttft_array = np.array(ttft_list) * 1000.0  # Convert s to ms
                            logger.info("\n{:{c}^{n}}".format(" TTFT Statistics ", c="=", n=50))
                            logger.info(f"Mean TTFT: {np.mean(ttft_array):.2f} ms")
                            logger.info(f"Median TTFT: {np.median(ttft_array):.2f} ms")
                            logger.info(f"P99 TTFT: {np.percentile(ttft_array, 99):.2f} ms")
                            logger.info(f"Min TTFT: {np.min(ttft_array):.2f} ms")
                            logger.info(f"Max TTFT: {np.max(ttft_array):.2f} ms")

                        if prefill_lat_list:
                            prefill_array = np.array(prefill_lat_list)  # Already in ms
                            logger.info("\n{:{c}^{n}}".format(" Prefill Latency Statistics ", c="=", n=50))
                            logger.info(f"Mean Prefill: {np.mean(prefill_array):.2f} ms")
                            logger.info(f"Median Prefill: {np.median(prefill_array):.2f} ms")
                            logger.info(f"P99 Prefill: {np.percentile(prefill_array, 99):.2f} ms")
                            logger.info(f"Min Prefill: {np.min(prefill_array):.2f} ms")
                            logger.info(f"Max Prefill: {np.max(prefill_array):.2f} ms")



                        if decode_lat_list:
                            decode_array = np.array(decode_lat_list)  # Already in ms
                            logger.info("\n{:{c}^{n}}".format(" Decode Latency Statistics ", c="=", n=50))
                            logger.info(f"Mean Decode: {np.mean(decode_array):.2f} ms")
                            logger.info(f"Median Decode: {np.median(decode_array):.2f} ms")
                            logger.info(f"P99 Decode: {np.percentile(decode_array, 99):.2f} ms")
                            logger.info(f"Min Decode: {np.min(decode_array):.2f} ms")
                            logger.info(f"Max Decode: {np.max(decode_array):.2f} ms")

                            # Calculate per-token decode latency statistics
                            # Decode latency is total GPU time for all output tokens per request
                            # Divide by number of output tokens per request to get per-token latency
                            if total_output_tokens > 0:
                                tokens_per_request = total_output_tokens / batch_size
                                per_token_decode_array = decode_array / tokens_per_request
                                logger.info("\n{:{c}^{n}}".format(" Per-Token Decode Latency Statistics ", c="=", n=50))
                                logger.info(f"Mean Per-Token Decode: {np.mean(per_token_decode_array):.2f} ms")
                                logger.info(f"Median Per-Token Decode: {np.median(per_token_decode_array):.2f} ms")
                                logger.info(f"P99 Per-Token Decode: {np.percentile(per_token_decode_array, 99):.2f} ms")
                                logger.info(f"Min Per-Token Decode: {np.min(per_token_decode_array):.2f} ms")
                                logger.info(f"Max Per-Token Decode: {np.max(per_token_decode_array):.2f} ms")

                        if itl_list:
                            itl_array = np.array(itl_list) * 1000.0  # Convert s to ms
                            logger.info("\n{:{c}^{n}}".format(" Inter-Token Latency Statistics ", c="=", n=50))
                            logger.info(f"Mean ITL: {np.mean(itl_array):.2f} ms")
                            logger.info(f"Median ITL: {np.median(itl_array):.2f} ms")
                            logger.info(f"P99 ITL: {np.percentile(itl_array, 99):.2f} ms")
                            logger.info(f"Min ITL: {np.min(itl_array):.2f} ms")
                            logger.info(f"Max ITL: {np.max(itl_array):.2f} ms")

                        if tpot_list:
                            tpot_array = np.array(tpot_list) * 1000.0  # Convert s to ms
                            logger.info("\n{:{c}^{n}}".format(" TPOT (Time Per Output Token) Statistics ", c="=", n=50))
                            logger.info(f"Mean TPOT: {np.mean(tpot_array):.2f} ms")
                            logger.info(f"Median TPOT: {np.median(tpot_array):.2f} ms")
                            logger.info(f"P99 TPOT: {np.percentile(tpot_array, 99):.2f} ms")
                            logger.info(f"Min TPOT: {np.min(tpot_array):.2f} ms")
                            logger.info(f"Max TPOT: {np.max(tpot_array):.2f} ms")

                        # Clear the lists for next iteration
                        metrics_collector.time_to_first_token_list.clear()
                        metrics_collector.inter_token_latency_seconds_list.clear()
                        metrics_collector.tpot_list.clear()
                        metrics_collector.prefill_latency_list.clear()
                        metrics_collector.decode_latency_list.clear()

    finally:
        from sglang.srt.utils import kill_process_tree

        kill_process_tree(os.getpid(), include_parent=False)


def get_vocab_size(model_config_path: str) -> int:
    """
    Loads the vocabulary size from the model's config.json file.

    Args:
        model_config_path (str): Path to the model directory containing config.json.

    Returns:
        int: Vocabulary size.
    """
    with open(os.path.join(model_config_path, "config.json"), "r") as f:
        config = json.load(f)
    return config.get("vocab_size", 0)


def main(cfg, deploy_cfg, backend_cfg, optional_cfg, bszs, seq_lens, output_lens, warmup_iters, benchmark_iters):
    """
    Reads YAML configs, prepares SGLang command-line arguments, and launches benchmarking.

    Args:
        cfg: Base configuration (OmegaConf).
        deploy_cfg: Deployment configuration (OmegaConf).
        backend_cfg: Backend configuration (OmegaConf).
        optional_cfg: Optional configuration (OmegaConf).
        bszs (List[int]): List of batch sizes.
        seq_lens (List[int]): List of sequence lengths.
        warmup_iters (int): Number of warmup iterations.
        benchmark_iters (int): Number of benchmark iterations.
    """
    sglang_args = []

    def yaml_to_args(yaml_cfg):
        """
        Converts a YAML config dictionary to a list of command-line arguments.

        Args:
            yaml_cfg: Configuration dictionary.

        Returns:
            List[str]: List of command-line arguments.
        """
        args = []
        for k, v in yaml_cfg.items():
            if v is None:
                continue
            if isinstance(v, bool):
                if v:
                    args.append(f"--{k.replace('_', '-')}")
                continue
            args.append(f"--{k.replace('_', '-')}={v}")
        return args

    base_args = yaml_to_args(cfg)
    deploy_args = yaml_to_args(deploy_cfg)
    backend_args = yaml_to_args(backend_cfg)
    optional_args = yaml_to_args(optional_cfg)
    sglang_args = base_args + deploy_args + backend_args + optional_args

    vocab_size = get_vocab_size(cfg.get("model"))
    # we don't log the args here because SGLang will log them
    launch_inference(
        sglang_args,
        bszs,
        seq_lens,
        vocab_size,
        max_new_tokens=output_lens,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
    )


def load_config(config_file):
    """
    Loads a YAML configuration file using OmegaConf.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        OmegaConf: Loaded configuration object.
    """
    return OmegaConf.load(config_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark configuration")
    parser.add_argument(
        "--base",
        type=str,
        default="conf/base.yaml",
        help="Path to the default configuration file",
    )
    parser.add_argument(
        "--deploy",
        type=str,
        default="conf/deploy/tp8ep8.yaml",
        help="Path to the deployment configuration file",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="conf/backend/fa3.yaml",
        help="Path to the backend configuration file",
    )
    parser.add_argument(
        "--optional",
        type=str,
        default="conf/optional/default.yaml",
        help="Path to the optional configuration file",
    )
    parser.add_argument(
        "--bszs",
        type=str,
        default="1",
        help="Batch sizes to use for benchmarking (comma-separated)",
    )
    parser.add_argument(
        "--seq_lens",
        type=str,
        default="10",
        help="Sequence lengths to use for benchmarking (comma-separated)",
    )
    parser.add_argument(
        "--output-lens",
        type=str,
        default="2",
        help="Number of tokens to generate during benchmarking",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=10,
        help="Number of benchmark iterations",
    )
    args, overrides = parser.parse_known_args()
    default_cfg = load_config(args.base)
    deploy_cfg = load_config(args.deploy)
    backend_cfg = load_config(args.backend)
    optional_cfg = load_config(args.optional)
    if overrides:
        # this is to support command line overrides
        cli_cfg = OmegaConf.from_dotlist(overrides)
        default_cfg = OmegaConf.merge(default_cfg, cli_cfg.get("base", {}))
        deploy_cfg = OmegaConf.merge(deploy_cfg, cli_cfg.get("deploy", {}))
        backend_cfg = OmegaConf.merge(backend_cfg, cli_cfg.get("backend", {}))
        optional_cfg = OmegaConf.merge(optional_cfg, cli_cfg.get("optional", {}))
    bszs = [int(bsz) for bsz in args.bszs.split(",")]
    seq_lens = [int(seq) for seq in args.seq_lens.split(",")]
    output_lens = [int(ol) for ol in args.output_lens.split(",")]
    main(default_cfg, deploy_cfg, backend_cfg, optional_cfg, bszs, seq_lens, output_lens, args.warmup_iters, args.benchmark_iters)
