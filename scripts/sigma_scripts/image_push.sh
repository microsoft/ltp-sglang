device=$1
if [ -z "$device" ]; then
    echo "Usage: $0 <device> <tag>"
    echo "Device options: cuda, rocm"
    exit 1
fi
if [ "$device" != "cuda" ] && [ "$device" != "rocm" ]; then
    echo "Invalid device option. Use 'cuda' or 'rocm'."
    exit 1
fi
tag=$2
if [ -z "$tag" ]; then
    tag="latest"
fi

registry="sigmainference.azurecr.io/sglang"

docker tag sglang-inference-$device:$tag $registry/sglang-inference-$device:$tag
docker push $registry/sglang-inference-$device:$tag