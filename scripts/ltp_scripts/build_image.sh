#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script builds Docker images for the LTP inference environment.
# Usage: ./build_image.sh [tag]

# Check if a tag argument is provided; if not, default to "latest"
tag=${1:-latest}

# Get the directory of the current script
script_dir=$(dirname "$0")
# Set the project directory to two levels up from the script directory
project_dir="$script_dir/../.."

# Build the CUDA-based Docker image
docker build \
    --build-arg BASE_IMAGE=lmsysorg/sglang:v0.4.10.post2-cu126 \
    --build-arg SGLANG_SRC=$project_dir \
    -t sglang-inference-cuda:$tag \
    -f ./dev.cuda.dockerfile \
    $project_dir

# Build the ROCm-based Docker image
docker build \
    --build-arg BASE_IMAGE=lmsysorg/sglang:v0.4.10.post2-rocm630-mi30x-srt \
    --build-arg SGLANG_SRC=$project_dir \
    -t sglang-inference-rocm:$tag \
    -f ./dev.rocm.dockerfile \
    $project_dir
