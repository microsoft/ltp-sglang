# Use the specified base image
ARG BASE_IMAGE=lmsysorg/sglang:v0.4.10.post2-cu126
FROM ${BASE_IMAGE}

RUN apt-get update -y && \
    apt-get install -y git-lfs

# Set the working directory
WORKDIR /sgl-workspace/sglang

# Specify the source directory for SGLang
ARG SGLANG_SRC=./

# Copy SGLang source files to the workspace
COPY ${SGLANG_SRC} /sgl-workspace/sglang

# Install SGLang with all optional dependencies
RUN python3 -m pip install --upgrade pip && \
    pip install ./python[dev] && \
    pip install ./python[test_ltp] --no-build-isolation
