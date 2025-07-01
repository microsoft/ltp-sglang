# under this path

docker build -t sglang-inference-cuda:latest -f ./dev.rocm.dockerfile ../../../
docker build -t sglang-inference-rocm:latest -f ./dev.cuda.dockerfile ../../../