FROM rocm/sgl-dev:20250518-srt

WORKDIR /app

RUN pip install sglang-router

COPY ./sglang /sgl-workspace/sglang

CMD ["tail", "-f", "/dev/null"]
