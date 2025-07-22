import os
import asyncio
import logging
import hydra
import numpy as np
from omegaconf import DictConfig
from typing import List
import manager_patch
import tpworker_patch
import gate_patch
from sglang.srt.server_args import prepare_server_args
from sglang.srt.entrypoints.engine import _launch_subprocesses 
from sglang.srt.managers.io_struct import GenerateReqInput

logger = logging.getLogger(__name__)

def launch_inference(args: List, bszs: List, seq_lens: List , max_new_tokens: int = 1):
    server_args = prepare_server_args(args)
    logger.info(server_args)
    try:
        tokenizer_manager, _  = _launch_subprocesses(server_args=server_args)
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
    finally:
        from sglang.srt.utils import kill_process_tree
        kill_process_tree(os.getpid(), include_parent=False)


@hydra.main(config_path="../../conf", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    """
    Read yaml config and launch inference.
    """
    bszs, seq_lens, max_tokens_generated = cfg.run.get("bszs", [1]), cfg.run.get("seq_lens", [4096]), cfg.run.get("max_tokens_generated", 1)
    
    args = [
        f"--model={cfg.model}",
        f"--log-level={cfg.log_level}",
        f"--load-format={cfg.load_format}",
        f"--chunked-prefill-size={cfg.chunked_prefill_size}",
        f"--max-prefill-tokens={cfg.max_prefill_tokens}",
        f"--attention-backend={cfg.attention_backend}",
        f"--mem-fraction-static={cfg.mem_fraction_static}",
        f"--max-running-requests={cfg.max_running_requests}",
        f"--dist-init-addr={cfg.run.dist_init_addr}",
        f"--nnodes={cfg.run.nnodes}",
        f"--node-rank={cfg.run.node_rank}",
        f"--tensor-parallel-size={cfg.run.tensor_parallel_size}",
    ]

    if cfg.run.get("enable_ep_moe"): 
        args.append("--enable-ep-moe")
        args.append(f"--ep-size={cfg.run.expert_parallel_size}")
        if cfg.run.get("enable_deepep_moe") and cfg.run.get("nnodes") > 1:
            args.append("--enable-deepep-moe")
    if cfg.run.get("enable_dp_attention"):
        args.append("--enable-dp-attention")
        args.append(f"--data-parallel-size={cfg.run.data_parallel_size}")
        if cfg.run.get("enable_dp_lm_head"):
            args.append(f"--enable-dp-lm-head")
        if cfg.run.get("enable_deepep_moe"):
            args.append(f"--deepep-mode=normal")
        
    if cfg.get("trust_remote_code"): args.append("--trust-remote-code")
    if cfg.get("disable_radix_cache"): args.append("--disable-radix-cache")
    if cfg.get("disable_custom_all_reduce"): args.append("--disable-custom-all-reduce")
    if cfg.get("disable_cuda_graph"): args.append("--disable-cuda-graph")
    if cfg.get("disable_overlap_schedule"): args.append("--disable-overlap-schedule")
    if cfg.get("disable_chunked_prefix_cache"): args.append("--disable-chunked-prefix-cache")
    
    print(args)
    launch_inference(args, bszs, seq_lens, max_tokens_generated)
    
if __name__ == "__main__":
    main()

