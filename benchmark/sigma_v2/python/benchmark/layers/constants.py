warmup_iters = 100
benchmark_iters = 1000
device = "cuda:0"
results_dir = "./results/"

layer_configs = {
    "mla": "./conf/mla_config.json",
    "attn": "./conf/attn_config.json",
    "linear": "./conf/linear_config.json",
    "exp": "./conf/moe_config.json",
    "moe": "./conf/moe_config.json"
}

layer_fig_groups = {
    "mla": {
        "prefill": ["prefill_fa3", "prefill_flashinfer"],
        "decode": ["decode_flashinfer", "decode_fa3", "decode_flashmla"]
    }, 
    "exp": {
        "exp": ["fused_moe", "ep_moe", "normal_deepep_moe", "low_lat_deepep_moe"]
    },
    "linear": {
        "linear_rep": ["fp16_linear", "fp8_linear"],
        "linear_col": ["fp16_col_linear", "fp8_col_linear"],
        "linear_row": ["fp16_row_linear", "fp8_row_linear"]
    }
}
