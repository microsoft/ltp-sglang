#!/bin/bash

# Check if device argument is provided
device=$1
if [ -z "$device" ]; then
    echo "Usage: $0 <device> <tag> <registry_url>"
    echo "Device options: cuda, rocm"
    exit 1
fi

# Validate device option
if [[ "$device" != "cuda" && "$device" != "rocm" ]]; then
    echo "Invalid device option. Use 'cuda' or 'rocm'."
    exit 1
fi

# Check if tag argument is provided; if not, default to "latest"
tag=${2:-"latest"}

# Define the registry
registry=${3:-"sigmainference.azurecr.io/sglang"}

# Tag and push the Docker image
docker tag "sglang-inference-$device:$tag" "$registry/sglang-inference-$device:$tag"
docker push "$registry/sglang-inference-$device:$tag"
