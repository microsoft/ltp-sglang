model_path=$1
if [ -z "$model_path" ]; then
    echo "Usage: $0 <model_path> <tp_size> <enable_ep> <attention_backend>"
    echo "Example: $0 /path/to/model 1 true triton"
    echo "  Defaults: tp_size=1, enable_ep=true, attention_backend=triton"
    echo "  Supported attention backends: triton, fa3"
    exit 1
fi

tp_size=${2:-1}
enable_ep=${3:-true}
attention_backend=${4:-triton}
if [ "$enable_ep" == "true" ] ; then
    enable_ep="--enable-ep-moe"
else
    enable_ep=""
fi


python3 -m sglang.launch_server --model-path $model_path --trust-remote-code --chat-template sigma --attention-backend $attention_backend --tp-size $tp_size $enable_ep
