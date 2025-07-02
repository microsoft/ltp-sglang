ARG BASE_IMAGE=rocm/sgl-dev:20250518-srt
FROM ${BASE_IMAGE}

WORKDIR /app

RUN pip install sglang-router

ARG SGLANG_SRC=./
COPY ${SGLANG_SRC} /sgl-workspace/sglang

CMD ["/bin/bash"]
