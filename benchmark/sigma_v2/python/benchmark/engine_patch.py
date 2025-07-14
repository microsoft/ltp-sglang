import os
import logging
import multiprocessing as mp
from typing import Optional, Tuple, Dict
import sglang.srt.managers.scheduler as scheduler
from sglang.srt.openai_api.adapter import (
    guess_chat_template_name_from_model_path,
)
logger = logging.getLogger(__name__)

def apply_multireturn_to_launch():
    """
    Add return of the processess created in the _launch_subprocesses function.
    """
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.managers.tokenizer_manager import TokenizerManager
    from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
    from sglang.srt.managers.data_parallel_controller import (
        run_data_parallel_controller_process,
    )
    from sglang.srt.code_completion_parser import load_completion_template_for_openai_api
    from sglang.srt.openai_api.adapter import load_chat_template_for_openai_api
    from sglang.srt.utils import (
        configure_logger,
        prepare_model_and_tokenizer,
        launch_dummy_health_check_server,
    )
    import sglang.srt.entrypoints.engine as engine
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    def _launch_subprocesses(
        server_args: ServerArgs, port_args: Optional[PortArgs] = None
    ) -> Tuple[TokenizerManager, Dict]:
        """
        Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.
        """
        # Configure global environment
        configure_logger(server_args)
        server_args.check_server_args()
        _set_envs_and_config(server_args)

        # Allocate ports for inter-process communications
        if port_args is None:
            port_args = PortArgs.init_new(server_args)
            logger.info(f"{server_args=}")

        # If using model from www.modelscope.cn, first download the model.
        server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
            server_args.model_path, server_args.tokenizer_path
        )

        scheduler_procs = []
        if server_args.dp_size == 1:
            # Launch tensor parallel scheduler processes
            memory_saver_adapter = TorchMemorySaverAdapter.create(
                enable=server_args.enable_memory_saver
            )

            scheduler_pipe_readers = []

            nnodes_per_tp_group = max(server_args.nnodes // server_args.pp_size, 1)
            tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
            tp_rank_range = range(
                tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
                tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
            )

            pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
            pp_rank_range = range(
                pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group),
                pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group + 1),
            )

            for pp_rank in pp_rank_range:
                for tp_rank in tp_rank_range:
                    reader, writer = mp.Pipe(duplex=False)
                    gpu_id = (
                        server_args.base_gpu_id
                        + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                        + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                    )
                    proc = mp.Process(
                        target=scheduler.run_scheduler_process,
                        args=(
                            server_args,
                            port_args,
                            gpu_id,
                            tp_rank,
                            pp_rank,
                            None,
                            writer,
                        ),
                    )
                    with memory_saver_adapter.configure_subprocess():
                        proc.start()
                    scheduler_procs.append(proc)
                    scheduler_pipe_readers.append(reader)
        else:
            # Launch the data parallel controller
            reader, writer = mp.Pipe(duplex=False)
            scheduler_pipe_readers = [reader]
            proc = mp.Process(
                target=run_data_parallel_controller_process,
                args=(server_args, port_args, writer),
            )
            proc.start()
            scheduler_procs.append(proc)

        if server_args.node_rank >= 1:
            # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
            # so they can just wait here.

            for reader in scheduler_pipe_readers:
                data = reader.recv()
                assert data["status"] == "ready"

            if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
                # When using `Engine` as a Python API, we don't want to block here.
                return None, None, None, None

            launch_dummy_health_check_server(server_args.host, server_args.port)

            for proc in scheduler_procs:
                proc.join()
                logger.error(
                    f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
                )
            return None, None, None, None

        # Launch detokenizer process
        detoken_proc = mp.Process(
            target=run_detokenizer_process,
            args=(
                server_args,
                port_args,
            ),
        )
        detoken_proc.start()

        # Launch tokenizer process
        tokenizer_manager = TokenizerManager(server_args, port_args)
        if server_args.chat_template:
            load_chat_template_for_openai_api(
                tokenizer_manager, server_args.chat_template, server_args.model_path
            )
        else:
            guess_chat_template_name_from_model_path(server_args.model_path)

        if server_args.completion_template:
            load_completion_template_for_openai_api(server_args.completion_template)

        # Wait for the model to finish loading
        scheduler_infos = []
        for i in range(len(scheduler_pipe_readers)):
            try:
                data = scheduler_pipe_readers[i].recv()
            except EOFError:
                logger.error(
                    f"Rank {i} scheduler is dead. Please check if there are relevant logs."
                )
                scheduler_procs[i].join()
                logger.error(f"Exit code: {scheduler_procs[i].exitcode}")
                raise

            if data["status"] != "ready":
                raise RuntimeError(
                    "Initialization failed. Please see the error messages above."
                )
            scheduler_infos.append(data)

        # Assume all schedulers have the same scheduler_info
        scheduler_info = scheduler_infos[0]
        tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
        return tokenizer_manager, scheduler_info, scheduler_procs, detoken_proc
    original_launch_subprocesses = engine._launch_subprocesses
    engine._launch_subprocesses = _launch_subprocesses

apply_multireturn_to_launch()
