#!/bin/bash  
   
model_path=$1
device=$2
# Check if model path argument is provided  
# Check if device argument is provided. It should be `cuda` or `rocm`.
if [ -z "$model_path" ] || [ -z "$device" ] || ([ "$device" != "cuda" ] && [ "$device" != "rocm" ]); then
    echo "Usage: $0 <model_path> <device> <image>"
    echo "Example: $0 /path/to/model cuda"
    echo "Supported devices: cuda, rocm"
    echo "Note: Ensure the model path is accessible and the device is correctly specified."
    exit 1
fi

# Set image to the third argument or default to the specified image
image=$3
if [ -z "$image" ]; then
    if [ "$device" == "cuda" ]; then
        image="sigmainference.azurecr.io/sglang/sglang-inference-cuda:latest"
    elif [ "$device" == "rocm" ]; then
        image="sigmainference.azurecr.io/sglang/sglang-inference-rocm:latest"
    fi
fi

# Run the Docker container based on the specified device
if [ "$device" == "cuda" ]; then
    echo "Running CUDA container with image: $image"    
    docker run -it \
        --name sigma-sglang \
        --gpus all \
        -p 30000:30000 \
        --ipc=host \
        --network=host \
        --privileged \
        -v $model_path:/app/sigma \
        $image \
        bash
elif [ "$device" == "rocm" ]; then
    echo "Running ROCm container with image: $image"
    docker run -it --rm \
        --name sigma-sglang \
        --ipc=host \
        --network=host \
        --privileged \
        -p 30000:30000 \
        --cap-add=CAP_SYS_ADMIN \
        --device=/dev/kfd \
        --device=/dev/dri \
        --device=/dev/mem \
        --group-add render \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v $model_path:/app/sigma \
        $image \
        bash
else
    echo "Unsupported device: $device"
    exit 1
fi
