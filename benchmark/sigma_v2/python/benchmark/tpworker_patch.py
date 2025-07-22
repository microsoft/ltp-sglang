import torch
import time
import logging
import threading
from typing import Optional, Tuple, Union
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)

WARMUP_STEPS = 50
RUN_STEPS = 50
PROFILE_STEPS = 50

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
    def run_multiple_times(self, forward_batch, pp_proxy_tensors=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i in range(WARMUP_STEPS + RUN_STEPS):
            if i == WARMUP_STEPS:
                start.record()
            logits_output, can_run_cuda_graph = self.model_runner.forward(forward_batch)
        end.record()
        torch.cuda.synchronize()
        print("Latency Benchmark Device {} {} {} {} {:.4f} ms".format(torch.cuda.current_device(), 
                                                                    convert_to_phase(forward_batch.forward_mode), 
                                                                    forward_batch.batch_size, 
                                                                    forward_batch.seq_lens[0], 
                                                                    start.elapsed_time(end) /RUN_STEPS))
        return logits_output, can_run_cuda_graph
    
    def run_multiple_times_profile(self, forward_batch, pp_proxy_tensors=None):
        torch.cuda.synchronize()
        for i in range(WARMUP_STEPS + PROFILE_STEPS):
            if i == WARMUP_STEPS + 25:
                torch.cuda.synchronize()
                if profile_prefill:
                    if forward_batch.forward_mode == ForwardMode.EXTEND:
                        torch.cuda.cudart().cudaProfilerStart()
                else:
                    if forward_batch.forward_mode == ForwardMode.DECODE:
                        torch.cuda.cudart().cudaProfilerStart()
            if i == WARMUP_STEPS + 26:
                torch.cuda.cudart().cudaProfilerStop()
            logits_output, can_run_cuda_graph = self.model_runner.forward(forward_batch)
        torch.cuda.synchronize()
        return logits_output, can_run_cuda_graph
    
    def forward_batch_generation(
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
            logits_output, can_run_cuda_graph = self.run_multiple_times(forward_batch, pp_proxy_tensors=None)
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
            logits_output, can_run_cuda_graph = self.run_multiple_times(forward_batch, pp_proxy_tensors=None)
            return pp_proxy_tensors.tensors, None, can_run_cuda_graph
    
    original_forward_batch_generation = TpModelWorker.forward_batch_generation
    TpModelWorker.forward_batch_generation = forward_batch_generation
    if profile:
        TpModelWorker.run_multiple_times = run_multiple_times_profile
    else:
        TpModelWorker.run_multiple_times = run_multiple_times
    

import util
profile, profile_phase = util.retrieve_profile_settings()
apply_timing_to_foward(profile, profile_phase)
