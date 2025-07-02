#!/bin/bash
script_dir=$(dirname "$0")
project_dir="$script_dir/../.."

docker build --build-arg BASE_IMAGE=lmsysorg/sglang:v0.4.6.post4-cu124 --build-arg SGLANG_SRC=$project_dir -t sglang-inference-cuda:latest -f ./dev.rocm.dockerfile $project_dir
docker build --build-arg BASE_IMAGE=rocm/sgl-dev:20250518-srt --build-arg SGLANG_SRC=$project_dir -t sglang-inference-rocm:latest -f ./dev.cuda.dockerfile $project_dir