import os
import argparse
import asyncio
import logging
import numpy as np
from typing import List
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

def launch_inference(args: List, bszs: List, seq_lens: List , max_new_tokens: int = 2):
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.entrypoints.engine import _launch_subprocesses 
    from sglang.srt.managers.io_struct import GenerateReqInput
    server_args = prepare_server_args(args)
    logger.info(server_args)
    try:
        tokenizer_manager, _, _  = _launch_subprocesses(server_args=server_args)
        async def runner():
            response = None
            async for item in tokenizer_manager.generate_request(request):
                response = item
            return response
        for batch_size in bszs:
            for seq in seq_lens:
                input_ids = [
                    [int(x) for x in np.random.randint(0, high=16384, size=(seq,))]
                    for _ in range(batch_size)
                ]
                request = GenerateReqInput(
                    input_ids = input_ids,
                    sampling_params ={
                        "temperature": 0,
                        "max_new_tokens": max_new_tokens,
                    }
                )
                ret = asyncio.get_event_loop().run_until_complete(runner())
                logger.info(f"Latency in Response: {ret[0]['meta_info']['e2e_latency']}")
    finally:
        from sglang.srt.utils import kill_process_tree
        kill_process_tree(os.getpid(), include_parent=False)

def main(cfg, deploy_cfg, backend_cfg, bszs, seq_lens):
    """
    Read yaml config and launch inference.
    """
    args = []
    
    if cfg.get("enable_benchmark"): args.append("--enable-benchmark")
    if cfg.get("enforce_batching"): args.append("--enforce-batching")
    if cfg.get("profile_phase"): args.append(f"--profile-phase={cfg.profile_phase}")
        
        
    args.extend([
        f"--model={cfg.model}",
        f"--load-format={cfg.load_format}",
        f"--log-level={cfg.log_level}",
        f"--max-prefill-tokens={cfg.max_prefill_tokens}",
        f"--chunked-prefill-size={cfg.chunked_prefill_size}",
        f"--mem-fraction-static={cfg.mem_fraction_static}",
        f"--max-running-requests={cfg.max_running_requests}",
    ])
    
    if cfg.get("trust_remote_code"): args.append("--trust-remote-code")
    if cfg.get("disable_radix_cache"): args.append("--disable-radix-cache")
    if cfg.get("disable_custom_all_reduce"): args.append("--disable-custom-all-reduce")
    if cfg.get("disable_cuda_graph"): args.append("--disable-cuda-graph")
    if cfg.get("disable_overlap_schedule"): args.append("--disable-overlap-schedule")
    if cfg.get("enable_torch_compile"): args.append("--enable-torch-compile")
    
    args.extend([
        f"--dist-init-addr={deploy_cfg.dist_init_addr}",
        f"--nnodes={deploy_cfg.nnodes}",
        f"--node-rank={deploy_cfg.node_rank}",
        f"--tensor-parallel-size={deploy_cfg.tensor_parallel_size}"
    ])

    if deploy_cfg.get("enable_ep_moe"): 
        args.extend([
            f"--ep-size={deploy_cfg.expert_parallel_size}"
        ])
        if deploy_cfg.get("enable_deepep_moe") and deploy_cfg.get("nnodes") > 1:
            args.append("--enable-deepep-moe")
    
    if deploy_cfg.get("enable_dp_attention"):
        args.extend([
            "--enable-dp-attention",
            f"--data-parallel-size={deploy_cfg.data_parallel_size}"
        ])
        if deploy_cfg.get("enable_dp_lm_head"): args.append(f"--enable-dp-lm-head")
        if deploy_cfg.get("enable_deepep_moe"): args.append(f"--deepep-mode=normal")
    
    args.append(f"--attention-backend={backend_cfg.attention_backend}")
    launch_inference(args, bszs, seq_lens, max_new_tokens=cfg.get("max_new_tokens", 2))
    
def load_config(config_file):
    return OmegaConf.load(config_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark configuration")
    parser.add_argument("--base", type=str, default="conf/base.yaml", help=
        "Path to the default configuration file")
    parser.add_argument("--deploy", type=str, default="conf/deploy/tp8ep8.yaml", help=
        "Path to the deployment configuration file")
    parser.add_argument("--backend", type=str, default="conf/backend/fa3.yaml", help=
        "Path to the backend configuration file")
    parser.add_argument("--bszs", type=str, default="1", help=
        "Batch sizes to use for benchmarking (comma-separated)")
    parser.add_argument("--seq_lens", type=str, default="10", help=
        "Sequence lengths to use for benchmarking (comma-separated)")
    args, overrides = parser.parse_known_args()
    default_cfg = load_config(args.base)
    deploy_cfg = load_config(args.deploy)
    backend_cfg = load_config(args.backend)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        if "base" in cli_cfg:
            default_cfg = OmegaConf.merge(default_cfg, cli_cfg.base)
        if "deploy" in cli_cfg:
            deploy_cfg = OmegaConf.merge(deploy_cfg, cli_cfg.deploy)
        if "backend" in cli_cfg:
            backend_cfg = OmegaConf.merge(backend_cfg, cli_cfg.backend)
    bszs = [int(bsz) for bsz in args.bszs.split(",")]
    seq_lens = [int(seq) for seq in args.seq_lens.split(",")]
    main(default_cfg, deploy_cfg, backend_cfg, bszs, seq_lens)