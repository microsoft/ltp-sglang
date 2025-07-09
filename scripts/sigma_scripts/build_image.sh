#!/bin/bash

# This script builds Docker images for the Sigma inference environment.
# Usage: ./build_image.sh [tag]

# Check if a tag argument is provided; if not, default to "latest"
tag=${1:-latest}

# Get the directory of the current script
script_dir=$(dirname "$0")
# Set the project directory to two levels up from the script directory
project_dir="$script_dir/../.."

# Build the CUDA-based Docker image
docker build \
    --build-arg BASE_IMAGE=lmsysorg/sglang:v0.4.6.post4-cu124 \
    --build-arg SGLANG_SRC=$project_dir \
    -t sglang-inference-cuda:$tag \
    -f ./dev.rocm.dockerfile \
    $project_dir

# Build the ROCm-based Docker image
docker build \
    --build-arg BASE_IMAGE=rocm/sgl-dev:20250518-srt \
    --build-arg SGLANG_SRC=$project_dir \
    -t sglang-inference-rocm:$tag \
    -f ./dev.cuda.dockerfile \
    $project_dir
