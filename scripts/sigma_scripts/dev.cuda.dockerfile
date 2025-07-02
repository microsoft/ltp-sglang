ARG BASE_IMAGE=lmsysorg/sglang:v0.4.6.post4-cu124
FROM ${BASE_IMAGE}

WORKDIR /app

ARG SGLANG_SRC=./
COPY ${SGLANG_SRC} /sgl-workspace/sglang

RUN pip install vllm==0.8.5
RUN cd /sgl-workspace/sglang && pip install -e "python[all]"
RUN pip install sglang-router

CMD ["bash"]
