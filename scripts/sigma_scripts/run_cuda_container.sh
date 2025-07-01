model_path=$1
if [ -z "$model_path" ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi
image=$2
if [ -z "$image" ]; then
    image="sigmainference.azurecr.io/sglang/sglang-inference-cuda:latest"
fi

docker run -it \
    --name dev-env \
    --gpus all \
    -p 30000:30000 \
    --ipc=host \
    -v $model_path:/app/sigma \
    $image \
    bash 
    # python3 -m sglang.launch_server --model-path /app/sigma --trust-remote-code --chat-template sigma --attention-backend triton 