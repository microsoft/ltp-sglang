# Set the base image
ARG BASE_IMAGE=rocm/sgl-dev:20250518-srt
FROM ${BASE_IMAGE}

# Set the working directory
WORKDIR /app

# Install the SGLang router package
RUN pip install sglang-router

# Specify the source directory for SGLang
ARG SGLANG_SRC=./

# Copy SGLang source files to the workspace
COPY ${SGLANG_SRC} /sgl-workspace/sglang
