#!/bin/bash
if [[ "$1" == "-h" ]]; then
    echo "Usage: $0 model(deepseek) ip(eth0 ip) nnodes(1) rank(0) tp(8) mconf(layer) dp(0) ep(0) deepep(false) wocg(false) profile(false) clock(1980)"
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            model="$2"
            shift 2
            ;;
        --ip)
            ip="$2"
            shift 2
            ;;
        --nnodes)
            nnodes="$2"
            shift 2
            ;;
        --rank)
            rank="$2"
            shift 2
            ;;
        --tp)
            tp="$2"
            shift 2
            ;;
        --mconf)
            mconf="$2"
            shift 2
            ;;
        --dp)
            dp="$2"
            shift 2
            ;;
        --ep)
            ep="$2"
            shift 2
            ;;
        --deepep)
            deepep="$2"
            shift 2
            ;;
        --wocg)
            cg="$2"
            shift 2
            ;;
        --profile)
            profile="$2"
            shift 2
            ;;
        --clock)
            clock="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

model=${model:-qwen_sigma}  # Default: qwen_sigma model
nnodes=${nnodes:-1}       # Default: 1 node
rank=${rank:-0}           # Default: rank 0
tp=${tp:-8}               # Default: tensor parallel size 8
mconf=${mconf:-layer}     # Default: one layer model config
dp=${dp:-0}               # Default: enable_dp_attention false
ep=${ep:-0}               # Default: enable_ep_moe false
deepep=${deepep:-false}   # Default: enable_deepep_moe false
wocg=${cg:-false}         # Default: disable cuda graph false
profile=${profile:-false} # Default: profiling disabled
gpuclock=${clock:-1980}  # Default: GPU clock speed 1980 MHz

local_ip=$(ip -4 addr show eth0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
if (( nnodes > 1 )); then
    ip=${ip:-10.0.0.10}
else
    ip=${ip:-$local_ip}
fi

enable_ep="$([ "$ep" -gt 0 ] && echo "true" || echo "false")"
enable_dp="$([ "$dp" -gt 0 ] && echo "true" || echo "false")"

run_script="python/benchmark/benchmark.py"
base_configs="conf/default.yaml"
deploy_configs="conf/run/deploy.yaml"
if [ ! -f "$base_configs" ]; then
    echo "Error: Base configuration file $base_configs not found."
    exit 1
fi
if [ ! -f "$deploy_configs" ]; then
    echo "Error: Deployment configuration file $deploy_configs not found."
    exit 1
fi
bsz_seq="scripts/bsz_seq.csv"
mkdir -p results/raw_output
profile_folder="${model}_profile_${gpuclock}/nsys_report"
IFS=$';'

declare -A mconfigs=(
    [layer]="model_conf/$model/config_layer.json"
    [full]="model_conf/$model/config_full.json"
)

if [[ -n "${mconfigs[$mconf]}" ]]; then
    target_mconf=${mconfigs[$mconf]}
else
    echo "Error: Invalid model config. Please choose from [layer, full]"
    exit 1
fi

echo "Using model: $target_mconf"
echo "Using deployment config: $deploy_configs"

if ! cp "$target_mconf" model_conf/$model/config.json; then
    echo "Error: Failed to copy $target_mconf to model_conf/config.json."
    exit 1
fi

update_config() {
    local key=$1
    local value=$2
    sed -i "s/${key}: .*/${key}: [$value]/" "$deploy_configs" || {
        echo "Error: Failed to update $key in $deploy_configs"
        exit 1
    }
}

run_params=(
    model="./model_conf/$model"
    disable_cuda_graph="$wocg"
    run="deploy"
    run.dist_init_addr="$ip:30000"
    run.nnodes="$nnodes"
    run.node_rank="$rank"
    run.tensor_parallel_size="$tp"
    run.enable_dp_attention="$enable_dp"
    run.data_parallel_size="$dp"
    run.enable_ep_moe="$enable_ep"
    run.expert_parallel_size="$ep"
    run.enable_deepep_moe="$deepep"
)

run_benchmark() {
    local enable_profile=$1
    local profile_phase=$2
    local bsz=$3
    local seq_len=$4
    
    sed -i "s/enable_profile: .*/enable_profile: $enable_profile /" "$base_configs" || {
        echo "Error: Failed to update profile setting in $base_configs"
        return 1
    }
    
    if [ "$profile" = "true" ]; then
        # Update profile phase configuration
        sed -i "s/profile_phase: .*/profile_phase: $profile_phase /" "$base_configs" || {
            echo "Error: Failed to update profile phase in $base_configs"
            return 1
        }
        sub_profile_folder="${profile_phase}_profile"
        # Create profile directory and set filename
        mkdir -p "$profile_folder"
        mkdir -p "${profile_folder}/${sub_profile_folder}"
        profile_filename="${profile_folder}/${sub_profile_folder}/profile_${profile_phase}_bsz_${bsz}_seq_${seq_len}_config_${mconf}"
        
        # Run with nsys profiling if file doesn't exist
        if [ -f "${profile_filename}.sqlite" ]; then
            echo "Profile already exists: ${profile_filename}.sqlite. Skipping nsys capture."
        else
            echo "Running profiling for phase: $profile_phase"
            SGL_ENABLE_JIT_DEEPGEMM=1 nsys profile \
                -t nvtx,cuda \
                --capture-range=cudaProfilerApi \
                --cuda-graph-trace node \
                -o "$profile_filename" \
                --export sqlite \
                --force-overwrite true \
                --cuda-memory-usage true \
                python3 "$run_script" "${run_params[@]}" || {
                    echo "Error: Profiling run failed"
                    return 1
                }
        fi
    else
        echo "Running latency benchmark"
        SGLANG_LOGGING_CONFIG_PATH="./scripts/log_config.json" SGL_ENABLE_JIT_DEEPGEMM=1 python3 "$run_script" "${run_params[@]}" || {
            echo "Error: Benchmark run failed"
            return 1
        }
    fi
    return 0
}

nvidia-smi -lgc "$gpuclock" || {
    echo "Error: Failed to set GPU clock speed to $gpuclock MHz."
    exit 1
} 

cleanup() {
    echo "Cleaning up processes..."
    sleep 10
    pkill -f sglang || true
    pkill -f "python.*$run_script" || true
    sleep 10
    # Force kill any remaining processes
    pkill -f -9 sglang 2>/dev/null || true
    pkill -f -9 "python.*$run_script" 2>/dev/null || true
    pkill -f -9 nsys 2>/dev/null || true
    sleep 10
}

while read -r bszs seq_lens; do
    echo "Running with batch sizes: $bszs, sequence lengths: $seq_lens"
    update_config "bszs" "$bszs"
    update_config "seq_lens" "$seq_lens"

    if [ ! -f "$run_script" ]; then
        echo "Error: Benchmark script $run_script not found."
        exit 1
    fi

    if [ "$profile" = "true" ]; then
        run_benchmark "true" "prefill" "$bszs" "$seq_lens"
        cleanup
        run_benchmark "true" "decode" "$bszs" "$seq_lens"
        cleanup
    else
        run_benchmark "false" "" "$bszs" "$seq_lens" 
    fi
done < "$bsz_seq"
