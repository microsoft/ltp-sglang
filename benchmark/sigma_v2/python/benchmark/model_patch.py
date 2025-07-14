import torch
import os
import logging
import threading
import torch.nn.functional as F
from functools import wraps
from typing import Optional, Tuple, Union
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.models.deepseek_v2 import MoEGate
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from paths import LOG_FILE

WARMUP_STEPS = 50
RUN_STEPS = 50
PROFILE_STEPS = 50
RECORD_LEVEL = 25

logging.addLevelName(RECORD_LEVEL, "RECORD")
logging.RECORD = RECORD_LEVEL
def record(self, message, *args, **kwargs):
    if args:
        self._log(RECORD_LEVEL, message, args, **kwargs)
    else:
        self._log(RECORD_LEVEL, message, (), **kwargs)
logging.Logger.record = record
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.RECORD)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

forward_mode_mapping = {
    1: "Prefill",
    2: "Decode"
}

def convert_to_phase(forward_mode: int) -> str:
    return forward_mode_mapping.get(forward_mode, "unknown")

def apply_timing_to_foward(profile: bool = False, profile_prefill: bool = False):
    """
    Patch the forward_batch_generation method to add timing and profiling to model forward.
    TODO profiling will cause some error
    """
    def run_multiple_times(model_runner, forward_batch, pp_proxy_tensors=None):
        for _ in range(WARMUP_STEPS):
            logits_output, can_run_cuda_graph = model_runner.forward(forward_batch)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(RUN_STEPS):
            logits_output, can_run_cuda_graph = model_runner.forward(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
        end.record()
        torch.cuda.synchronize()
        logger.record("Device {} {} Latency {} {} {:.4f} ms".format(torch.cuda.current_device(), 
                                                                    convert_to_phase(forward_batch.forward_mode), 
                                                                    forward_batch.batch_size, 
                                                                    forward_batch.seq_lens[0], 
                                                                    start.elapsed_time(end) /RUN_STEPS))
        return logits_output, can_run_cuda_graph
    
    def run_multiple_times_profile(model_runner, forward_batch, pp_proxy_tensors=None):
        for _ in range(WARMUP_STEPS):
            logits_output, can_run_cuda_graph = model_runner.forward(forward_batch)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(PROFILE_STEPS):
            if i == 25:
                torch.cuda.synchronize()
                if profile_prefill:
                    if forward_batch.forward_mode == 1:
                        torch.cuda.cudart().cudaProfilerStart()
                else:
                    if forward_batch.forward_mode == ForwardMode.DECODE:
                        torch.cuda.cudart().cudaProfilerStart()
            if i == 26:
                torch.cuda.cudart().cudaProfilerStop()
            logits_output, can_run_cuda_graph = model_runner.forward(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
        end.record()
        torch.cuda.synchronize()
        return logits_output, can_run_cuda_graph
    
    def forward_batch_generation_timing(
        self,
        model_worker_batch: ModelWorkerBatch,
        launch_done: Optional[threading.Event] = None,
        skip_sample: bool = False,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        pp_proxy_tensors = None
        if not self.pp_group.is_first_rank:
            pp_proxy_tensors = PPProxyTensors(
                self.pp_group.recv_tensor_dict(
                    all_gather_group=self.get_attention_tp_group()
                )
            )

        if self.pp_group.is_last_rank:
            logits_output, can_run_cuda_graph = run_multiple_times(self.model_runner, forward_batch, pp_proxy_tensors=pp_proxy_tensors)
            if launch_done is not None:
                launch_done.set()

            if skip_sample:
                next_token_ids = None
            else:
                next_token_ids = self.model_runner.sample(
                    logits_output, model_worker_batch
                )

            return logits_output, next_token_ids, can_run_cuda_graph
        else:
            logits_output, can_run_cuda_graph = run_multiple_times(self.model_runner, forward_batch, pp_proxy_tensors=pp_proxy_tensors)
            return pp_proxy_tensors.tensors, None, can_run_cuda_graph

    def forward_batch_generation_profile(
        self,
        model_worker_batch: ModelWorkerBatch,
        launch_done: Optional[threading.Event] = None,
        skip_sample: bool = False,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        pp_proxy_tensors = None
        if not self.pp_group.is_first_rank:
            pp_proxy_tensors = PPProxyTensors(
                self.pp_group.recv_tensor_dict(
                    all_gather_group=self.get_attention_tp_group()
                )
            )
            
        if self.pp_group.is_last_rank:
            logits_output, can_run_cuda_graph = run_multiple_times_profile(self.model_runner, forward_batch, pp_proxy_tensors=pp_proxy_tensors)
            if launch_done is not None:
                launch_done.set()

            if skip_sample:
                next_token_ids = None
            else:
                next_token_ids = self.model_runner.sample(
                    logits_output, model_worker_batch
                )

            return logits_output, next_token_ids, can_run_cuda_graph
        else:
            logits_output, can_run_cuda_graph = run_multiple_times_profile(self.model_runner, forward_batch, pp_proxy_tensors=pp_proxy_tensors)
            return pp_proxy_tensors.tensors, None, can_run_cuda_graph
    
    original_forward_batch_generation = TpModelWorker.forward_batch_generation
    TpModelWorker.forward_batch_generation = forward_batch_generation_timing
    if profile:
        TpModelWorker.forward_batch_generation = forward_batch_generation_profile

import util
profile, profile_phase = util.retrieve_profile_settings()
apply_timing_to_foward(profile, profile_phase)
#apply_timing_to_foward(True, True)
