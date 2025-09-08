#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Check if device argument is provided
device=$1
# Define the registry
registry=$2
if [ -z "$device" ] || [ -z "$registry" ]; then
    echo "Usage: $0 <device> <registry_url> <tag>"
    echo "Device options: cuda, rocm"
    exit 1
fi

# Validate device option
if [[ "$device" != "cuda" && "$device" != "rocm" ]]; then
    echo "Invalid device option. Use 'cuda' or 'rocm'."
    exit 1
fi

# Check if tag argument is provided; if not, default to "latest"
tag=${3:-"latest"}

# Tag and push the Docker image
docker tag "sglang-inference-$device:$tag" "$registry/sglang-inference-$device:$tag"
docker push "$registry/sglang-inference-$device:$tag"
