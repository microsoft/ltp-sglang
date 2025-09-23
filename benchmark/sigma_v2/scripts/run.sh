#!/bin/bash
set -e  # Exit on error

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --base BASE         Base configuration (default: base)"
    echo "  --deploy DEPLOY     Deployment configuration (default: tp8ep8)"
    echo "  --backend BACKEND   Backend configuration (default: fa3)"
    echo "  --optional OPTIONAL Optional other configurations (default: default)"
    echo "  --model MODEL       Model to use (default: qwen_sigma)"
    echo "  --mconf MODELCONF   Model conf to use (default: full)"
    echo "  --ip IP             IP address (default: eth0 IP or 10.0.0.101)"
    echo "  --rank RANK         Rank (default: 0)"
    echo "  --save SAVE         Save results to a specific directory (default: false)"
    echo "  --profile TYPE      Profiling type: none, prefill, decode, both (default: none)"
    echo "  -h, --help          Display this help message"
    exit 1
}

cleanup() {
    echo "Cleaning up processes..."
    sleep 5
    pkill -f sglang 2>/dev/null || true
    pkill -f "python.*$run_script" 2>/dev/null || true
    sleep 5
    pkill -f -9 sglang 2>/dev/null || true
    pkill -f -9 "python.*$run_script" 2>/dev/null || true
    pkill -f -9 nsys 2>/dev/null || true
    sleep 5
}

run_benchmark() {
    local profile_phase=$1
    local bsz=$2
    local seq_len=$3

    if [[ -n "$profile_phase" ]]; then
        echo "Running profiling $profile_phase phase"
        local profile_folder="results/${model}_profile/nsys_report"
        local sub_profile_folder="${profile_phase}_profile"
        mkdir -p "$profile_folder/${sub_profile_folder}"
        local profile_filename="${profile_folder}/${sub_profile_folder}/profile_${profile_phase}_bsz_${bsz}_seq_${seq_len}_base_${base}_config_${deploy}_backend_${backend}"

        local run_params_profile=("${run_params[@]}" "base.profile_phase=$profile_phase")
        nsys profile \
            -t nvtx,cuda \
            --capture-range=cudaProfilerApi \
            --cuda-graph-trace node \
            -o "$profile_filename" \
            --export sqlite \
            --force-overwrite true \
            python3 "$run_script" "${run_params_profile[@]}" || {
                echo "Error: Profiling run failed"
                return 1
            }
    else
        echo "Running latency benchmark"
        python3 -u "$run_script" "${run_params[@]}" 2>&1 | tee -a "$benchmark_log_filename"
    fi
    return 0
}

# Check for help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base)
            base="$2"
            shift 2
            ;;
        --deploy)
            deploy="$2"
            shift 2
            ;;
        --backend)
            backend="$2"
            shift 2
            ;;
        --optional)
            optional="$2"
            shift 2
            ;;
        --model)
            model="$2"
            shift 2
            ;;
        --mconf)
            mconf="$2"
            shift 2
            ;;
        --ip)
            ip="$2"
            shift 2
            ;;
        --rank)
            rank="$2"
            shift 2
            ;;
        --save)
            save="$2"
            shift 2
            ;;
        --profile)
            profile="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

base="${base:-base}"          # Default: default config
deploy="${deploy:-tp8ep8}"    # Default: tp8ep8 deployment config
backend="${backend:-fa3}"     # Default: fa3 backend config
optional="${optional:-default}"  # Default: default optional config
model="${model:-qwen_sigma}"  # Default: qwen_sigma model
mconf="${mconf:-full}"        # Default: full model config
rank="${rank:-0}"             # Default: rank 0
save="${save:-false}"         # Default: do not save results
profile="${profile:-none}"    # Default: profiling disabled


if [[ -z "${ip:-}" ]]; then
    if [[ $deploy != *"tp8"* ]]; then
        ip="10.0.0.101"
    else
        ip=$(ip -4 addr show eth0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || echo "")
    fi
fi

benchmark_log_filename="results/raw_output/benchmark_$(date +%Y%m%d_%H%M%S).log"
run_script="python/benchmark/benchmark.py"
analysis_script="python/tools/analysis.py"
base_config="conf/$base.yaml"
deploy_config="conf/deploy/$deploy.yaml"
backend_config="conf/backend/$backend.yaml"
optional_config="conf/optional/$optional.yaml"
model_config="model_conf/$model"
target_model_config="${model_config}/config_${mconf}.json"

bsz_seq="scripts/bsz_seq.csv"

for file in "$base_config" "$deploy_config" "$backend_config" "$optional_config" "$run_script" "$bsz_seq" "$target_model_config"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: Required file $file not found."
        exit 1
    fi
done

cp "$target_model_config" "$model_config/config.json"

mkdir -p results/raw_output

run_params=(
    "--base" "$base_config"
    "--deploy" "$deploy_config"
    "--backend" "$backend_config"
    "--optional" "$optional_config"
    "base.model=$model_config"
    "deploy.dist_init_addr=$ip:30000"
    "deploy.node_rank=$rank"
)

max_gr=$(nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader,nounits -i 0)
nvidia-smi -lgc $max_gr || echo "Warning: Could not set GPU clock"

echo "Using model config: $model $mconf" | tee -a "$benchmark_log_filename"
echo "Using configs: $base, $deploy, $backend, $optional" | tee -a "$benchmark_log_filename"

IFS=$';'
while read -r bszs seq_lens; do
    echo "Running with batch sizes: $bszs, sequence lengths: $seq_lens" | tee -a "$benchmark_log_filename"
    run_params+=("--bszs=$bszs" "--seq_lens=$seq_lens")
    if [[ "$profile" == "none" ]]; then
        run_benchmark "" "$bszs" "$seq_lens"
        if [[ "$save" == "true" ]]; then
            python3 $analysis_script --log "$benchmark_log_filename"
        fi
    elif [[ "$profile" == "prefill" ||  "$profile" == "both" ]]; then
        run_benchmark "prefill" "$bszs" "$seq_lens"
        cleanup
    fi

    if [[ "$profile" == "decode" || "$profile" == "both" ]]; then
        run_benchmark "decode" "$bszs" "$seq_lens"
        cleanup
    fi
done < "$bsz_seq"
