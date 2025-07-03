model_path=$1
if [ -z "$model_path" ]; then
    echo "Usage: $0 <model_path> <dp_size> <tp_size> <attention_backend> <bench_backend>"
    echo "Example: $0 /path/to/model 1 1 triton sglang"
    echo "  Defaults: dp_size=1, tp_size=1, attention_backend=triton, bench_backend=sglang"
    echo "  Supported attention backends: sigma-17b: triton, sigma-200b: h200: fa3 ..."
    echo "  Supported backends: {sglang, sglang-native, sglang-oai, vllm, lmdeploy, trt, gserver, truss}"
    exit 1
fi

dp_size=$2
if [ -z "$dp_size" ]; then
    dp_size=1
fi

tp_size=$3
if [ -z "$tp_size" ]; then
    tp_size=1
fi

attention_backend=$4 # {triton, flashinfer, torch_native}
if [ -z "$attention_backend" ]; then
    attention_backend="triton"
fi

bench_backend=$5 # {sglang,sglang-native,sglang-oai,vllm,lmdeploy,trt,gserver,truss}
if [ -z "$bench_backend" ]; then
    bench_backend="sglang"
fi

echo "Running sglang model path: $model_path, logging to ./sglang.log."
echo " Note: for benchmarking, the radix cache (KV cache reuse) is disabled (--disable-radix-cache) "

if [ $dp_size -gt 1 ]; then
    echo "Running with data parallelism size: $dp_size"
    python3 -m  sglang_router.launch_server --model-path $model_path --trust-remote-code --chat-template sigma --attention-backend $attention_backend --dp-size $dp_size --tp-size $tp_size  --disable-radix-cache > sglang.log 2>&1 &
else
    echo "Running without data parallelism."
    python3 -m sglang.launch_server --model-path $model_path --trust-remote-code --chat-template sigma --attention-backend $attention_backend --tp-size $tp_size --disable-radix-cache > sglang.log 2>&1 &
fi

echo "Waiting for server to start..."
date
sleep 100 # adjust as needed based on model size and hardware

echo "Running benchmarks with sglang-oai backend."
echo ">> Running offline benchmarks, logging to offline.jsonl..." 
python3 -m sglang.bench_serving --backend $bench_backend --dataset-name random --num-prompts 2000 --random-input 1024 --random-output 1024 --output-file offline-dp$dp_size-tp$tp_size.jsonl
python3 -m sglang.bench_serving --backend $bench_backend --dataset-name random --num-prompts 2000 --random-input 1024 --random-output 512 --output-file offline-dp$dp_size-tp$tp_size.jsonl
python3 -m sglang.bench_serving --backend $bench_backend --dataset-name random --num-prompts 2000 --random-input 2048 --random-output 512 --output-file offline-dp$dp_size-tp$tp_size.jsonl
python3 -m sglang.bench_serving --backend $bench_backend --dataset-name sharegpt --num-prompts 2000 --output-file offline-dp$dp_size-tp$tp_size.jsonl

echo ">> Running online benchmarks, logging to online.jsonl..."
python3 -m sglang.bench_serving --backend $bench_backend --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 300 --request-rate 1 --output-file online-dp$dp_size-tp$tp_size.jsonl
python3 -m sglang.bench_serving --backend $bench_backend --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 600 --request-rate 2 --output-file online-dp$dp_size-tp$tp_size.jsonl
python3 -m sglang.bench_serving --backend $bench_backend --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 1200 --request-rate 4 --output-file online-dp$dp_size-tp$tp_size.jsonl
python3 -m sglang.bench_serving --backend $bench_backend --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 2400 --request-rate 8 --output-file online-dp$dp_size-tp$tp_size.jsonl
python3 -m sglang.bench_serving --backend $bench_backend --dataset-name random --random-input 1024 --random-output 1024 --num-prompts 3200 --request-rate 16 --output-file online-dp$dp_size-tp$tp_size.jsonl
