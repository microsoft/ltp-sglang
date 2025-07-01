FROM lmsysorg/sglang:v0.4.6.post4-cu124

WORKDIR /app

COPY ./sglang /sgl-workspace/sglang

RUN pip install vllm==0.8.5
RUN cd /sgl-workspace/sglang && pip install -e "python[all]"
RUN pip install sglang-router

CMD ["tail", "-f", "/dev/null"]
