import torch
import logging
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.utils import log_info_on_rank0, is_cuda

_is_cuda = is_cuda()
logger = logging.getLogger(__name__)

def determine_n_share_experts_fusion(
    self, architecture: str = "DeepseekV3ForCausalLM"
):
    self.n_share_experts_fusion = global_server_args_dict["n_share_experts_fusion"]
    if self.n_share_experts_fusion > 0:
        # Only Deepseek V3/R1 can use shared experts fusion optimization now.
        if (
            not _is_cuda
            or self.config.architectures[0] != architecture
            or self.config.n_routed_experts != 256
        ):
            self.n_share_experts_fusion = 0
            global_server_args_dict["n_share_experts_fusion"] = 0
            log_info_on_rank0(
                logger,
                "Only Deepseek V3/R1 on NV-platform can use shared experts fusion optimization. Shared experts fusion optimization is disabled.",
            )
        else:
            assert (
                self.n_share_experts_fusion == self.tp_size
            ), f"Shared experts fusion optimization is enabled in DeepSeek V3/R1, set it to {self.tp_size} can get best optimized performance."
    elif self.n_share_experts_fusion == 0:
        if (
            _is_cuda
            and torch.cuda.get_device_capability("cuda") >= (9, 0)
            and self.config.architectures[0] == architecture
            and self.config.n_routed_experts == 256
            and (not global_server_args_dict["enable_deepep_moe"])
        ):
            self.n_share_experts_fusion = 0
            global_server_args_dict["n_share_experts_fusion"] = 0
            log_info_on_rank0(
                logger,
                "Deepseek V3/R1 with fp8 can use shared experts fusion optimization when SM version >=90. Shared experts fusion optimization is enabled.",
            )

DeepseekV2ForCausalLM.determine_n_share_experts_fusion = determine_n_share_experts_fusion