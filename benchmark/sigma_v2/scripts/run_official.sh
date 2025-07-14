SGL_ENABLE_JIT_DEEPGEMM=1 python3 -m sglang.launch_server --model-path model_conf/deepseek --tp 16 --trust-remote-code --load-format=dummy --disable-radix-cache --enable-ep-moe --ep-size 16 --enable-deepep-moe --nnodes 2 --node-rank 0 --dist-init-addr 10.0.0.10:30000 --host 0.0.0.0 --port 2000 --enable-dp-attention --data-parallel-size 16 --deepep-mode normal
SGL_ENABLE_JIT_DEEPGEMM=1 python3 -m sglang.launch_server --model-path model_conf/deepseek --tp 16 --trust-remote-code --load-format=dummy --disable-radix-cache --enable-ep-moe --ep-size 16 --enable-deepep-moe --nnodes 2 --node-rank 1 --dist-init-addr 10.0.0.10:30000 --enable-dp-attention --data-parallel-size 16 --deepep-mode normal

python3 -m sglang.bench_one_batch_server --model model_conf/deepseek --base-url http://10.0.0.10:2000 --batch-size 1 --input-len 128 --output-len 128

python3 -m sglang.bench_serving --backend sglang-oai --num-prompts 32 --request-rate 1 --dataset-name random --random-input-len 1024 --random-output-len 2 --random-range-ratio 1 --base-url http://10.0.0.10:2000

SGL_ENABLE_JIT_DEEPGEMM=1 python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --trust-remote-code --load-format=dummy --disable-radix-cache --nnodes 2 --node-rank 0 --dist-init-addr 10.0.0.10:30000 --host 0.0.0.0 --port 2000
SGL_ENABLE_JIT_DEEPGEMM=1 python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --trust-remote-code --load-format=dummy --disable-radix-cache --nnodes 2 --node-rank 1 --dist-init-addr 10.0.0.10:30000 

python -m sglang.bench_one_batch --model-path xai-org/grok-1 --load-format dummy  --batch 1 --input-len 256 