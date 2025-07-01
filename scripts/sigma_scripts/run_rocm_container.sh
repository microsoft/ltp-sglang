model_path=$1
if [ -z "$model_path" ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi
image=$2
if [ -z "$image" ]; then
    image="sigmainference.azurecr.io/sglang/sglang-inference-rocm:latest"
fi


docker run -it --rm \
    --name dev-env \
    --ipc=host \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --device=/dev/kfd \
    --device=/dev/dri \
    --device=/dev/mem \
    --group-add render \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $model_path:/app/sigma \
    $image \
    bash