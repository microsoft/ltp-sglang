import torch
import sglang.srt.distributed as sg_dist
def apply_hardcode_parallel(tp):
    sg_dist.get_tensor_model_parallel_world_size = lambda: tp
    sg_dist.get_tensor_model_parallel_rank = lambda: 0
import sglang.srt.utils as sg_utils
def dispose_tensor(x: torch.Tensor):
    pass
sg_utils.dispose_tensor = dispose_tensor