import os
import json
import argparse
import asyncio
import logging
import numpy as np
from typing import List
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

NON_SGLANG_CMDLINE_ARGS = ["max_tokens_generated"]

def launch_inference(args: List, bszs: List, seq_lens: List, vocab_size: int, max_new_tokens: int = 2):
    from sglang.srt.entrypoints.engine import _launch_subprocesses
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.server_args import prepare_server_args

    server_args = prepare_server_args(args)
    logger.info(server_args)
    try:
        tokenizer_manager, _, _ = _launch_subprocesses(server_args=server_args)

        async def runner():
            response = None
            async for item in tokenizer_manager.generate_request(request):
                response = item
            return response

        for batch_size in bszs:
            for seq in seq_lens:
                input_ids = [
                    [int(x) for x in np.random.randint(0, high=vocab_size, size=(seq,))]
                    for _ in range(batch_size)
                ]
                request = GenerateReqInput(
                    input_ids=input_ids,
                    sampling_params={
                        "temperature": 0,
                        "max_new_tokens": max_new_tokens,
                    },
                )
                ret = asyncio.get_event_loop().run_until_complete(runner())
                logger.info(
                    f"Latency in Response: {ret[0]['meta_info']['e2e_latency']}"
                )
    finally:
        from sglang.srt.utils import kill_process_tree

        kill_process_tree(os.getpid(), include_parent=False)


def get_vocab_size(model_config_path: str) -> int:
    with open(os.path.join(model_config_path, "config.json"), "r") as f:
        config = json.load(f)
    return config.get("vocab_size", 0)

def main(cfg, deploy_cfg, backend_cfg, optional_cfg, bszs, seq_lens):
    """
    Read yaml config and launch inference.
    """
    sglang_args = []

    def yaml_to_args(yaml_cfg):
        args = []
        for k, v in yaml_cfg.items():
            if v is None or k in NON_SGLANG_CMDLINE_ARGS:
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
    launch_inference(sglang_args, bszs, seq_lens, vocab_size, max_new_tokens=cfg.get("max_new_tokens", 2))


def load_config(config_file):
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
    main(default_cfg, deploy_cfg, backend_cfg, optional_cfg, bszs, seq_lens)
