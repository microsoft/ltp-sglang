#!/bin/bash

model_path=$1
model_name=$2
device=$3
image=$4
# Check if model path argument is provided
# Check if device argument is provided. It should be `cuda` or `rocm`.
if [ -z "$model_path" ] || [ -z "$model_name" ] || [ -z "$device" ] || ([ "$device" != "cuda" ] && [ "$device" != "rocm" ]); then
    echo "Usage: $0 <model_path> <model_name> <device> <image>"
    echo "Example: $0 /path/to/model your_model_name cuda"
    echo "Supported devices: cuda, rocm"
    echo "Note: Ensure the model path is accessible and the device is correctly specified."
    exit 1
fi

# Run the Docker container based on the specified device
if [ "$device" == "cuda" ]; then
    echo "Running CUDA container with image: $image"
    docker run -it \
        --name ltp-sglang \
        --gpus all \
        --ipc=host \
        --privileged \
        -v $model_path:/app/$model_name \
        $image \
        bash
elif [ "$device" == "rocm" ]; then
    echo "Running ROCm container with image: $image"
    docker run -it --rm \
        --name ltp-sglang \
        --ipc=host \
        --privileged \
        --cap-add=CAP_SYS_ADMIN \
        --device=/dev/kfd \
        --device=/dev/dri \
        --device=/dev/mem \
        --group-add render \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v $model_path:/app/$model_name \
        $image \
        bash
else
    echo "Unsupported device: $device"
    exit 1
fi
