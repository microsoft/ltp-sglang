# Use the specified base image
ARG BASE_IMAGE=lmsysorg/sglang:v0.4.6.post4-cu124
FROM ${BASE_IMAGE}

# Set the working directory
WORKDIR /app

# Specify the source directory for SGLang
ARG SGLANG_SRC=./

# Copy SGLang source files to the workspace
COPY ${SGLANG_SRC} /sgl-workspace/sglang

# Install the necessary Python packages
RUN pip install vllm==0.8.5

# Install SGLang with all optional dependencies
RUN cd /sgl-workspace/sglang && pip install -e "python[all]"

# Install the SGLang router
RUN pip install sglang-router
