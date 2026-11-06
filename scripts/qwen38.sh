#!/usr/bin/env bash
# One-command balanced Flash-Next serving on 4x V100 32 GB.
set -euo pipefail

q38_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
q38_repo_dir=$(cd -- "$q38_script_dir/.." && pwd)
q38_action=${1:-start}
if (( $# )); then shift; fi
q38_image=${QWEN38_IMAGE:-geesegeesegeese/sglang-v100:v100-qwen38-flash-next-v5}
q38_container=${QWEN38_CONTAINER:-qwen38}
q38_cache=${HF_HOME:-$HOME/.cache/huggingface}
q38_jit_volume=${QWEN38_JIT_VOLUME:-qwen38-v5-jit}
q38_port=${PORT:-8082}
q38_label=org.sglang.qwen38.managed

usage() {
    cat <<'EOF'
Usage: bash scripts/qwen38.sh [start|logs|status|stop|restart|recreate|pull|build|host]

  start     Start Docker in the background (default).
  logs      Follow startup and request logs.
  status    Show container and readiness status.
  stop      Stop Docker, preserving its container and caches.
  restart   Restart the existing container.
  recreate  Replace the container with the selected image; preserve caches.
  pull      Download the selected release image.
  build     Build the Docker image from this checkout.
  host      Run the same profile in the active host Python environment.

Defaults: port 8082, model qwen, native MTP-3/4, eight requests, 262K context.
Startup and restarts are manual; nothing is enabled for boot.
Set PORT, HF_HOME, QWEN38_CONTAINER or QWEN38_IMAGE when needed.
Extra arguments after start/recreate/host are passed to SGLang.
EOF
}

case "$q38_action" in
    help|-h|--help) usage; exit 0 ;;
    host)
        export PORT="$q38_port"
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
        export SGLANG_SM70_MTP_LOCAL_TOP1=0 SGLANG_SM70_HC_COMBINE_NORM=0 SGLANG_SM70_AR_PUSH16=0
        exec bash "$q38_script_dir/serve_qwen38_flash_next_nvfp4_v100_throughput.sh" \
            RadixArk/Qwen3.8-Flash-Next-NVFP4 "$@"
        ;;
    start|logs|status|stop|restart|recreate|pull|build) ;;
    *) usage >&2; exit 2 ;;
esac

command -v docker >/dev/null || { echo "Docker is required for this command." >&2; exit 1; }

if [[ $q38_action == start || $q38_action == recreate ]]; then
    if [[ ! $q38_port =~ ^[0-9]{1,5}$ ]] || (( 10#$q38_port < 1 || 10#$q38_port > 65535 )); then
        echo "PORT must be between 1 and 65535." >&2
        exit 2
    fi
fi

exists() { docker container inspect "$q38_container" >/dev/null 2>&1; }
managed() {
    [[ $(docker inspect --format "{{index .Config.Labels \"$q38_label\"}}" "$q38_container") == true ]] || {
        echo "Container $q38_container was not created by this launcher; leaving it alone." >&2
        exit 1
    }
}

start() {
    if exists; then
        managed
        echo "Using existing container $q38_container. Use recreate to apply a new image or settings."
        docker update --restart=no "$q38_container" >/dev/null
        docker start "$q38_container"
        return
    fi
    if (true <>"/dev/tcp/127.0.0.1/$q38_port") 2>/dev/null; then
        echo "Port $q38_port is in use. Stop the current server before starting Docker." >&2
        exit 1
    fi
    mkdir -p "$q38_cache"
    q38_env=()
    for q38_key in HF_TOKEN HF_HUB_OFFLINE; do
        if [[ -v $q38_key ]]; then q38_env+=(--env "$q38_key"); fi
    done
    docker run -d --name "$q38_container" --label "$q38_label=true" \
        --restart no --stop-timeout 60 \
        --gpus all --network host --ipc host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        --volume "$q38_cache:/root/.cache/huggingface" \
        --volume "$q38_jit_volume:/root/sglang-v100-jit" \
        --env "PORT=$q38_port" \
        --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
        "${q38_env[@]}" "$q38_image" \
        bash /opt/sglang/scripts/serve_qwen38_flash_next_nvfp4_v100_throughput.sh \
        RadixArk/Qwen3.8-Flash-Next-NVFP4 "$@"
    echo "Starting on http://127.0.0.1:$q38_port/v1 (model: qwen)."
    echo "Follow progress: bash $q38_script_dir/qwen38.sh logs"
}

case "$q38_action" in
    start) start "$@" ;;
    logs) managed; exec docker logs --follow --tail 80 "$q38_container" ;;
    status)
        managed
        exec docker inspect --format '{{.Name}}: {{.State.Status}} {{if .State.Health}}({{.State.Health.Status}}){{end}}' "$q38_container"
        ;;
    stop) managed; exec docker stop --time 60 "$q38_container" ;;
    restart)
        managed
        docker update --restart=no "$q38_container" >/dev/null
        exec docker restart --time 60 "$q38_container"
        ;;
    recreate)
        if exists; then
            managed
            docker stop --time 60 "$q38_container"
            docker rm "$q38_container"
        fi
        start "$@"
        ;;
    pull) exec docker pull "$q38_image" ;;
    build)
        q38_revision=$(git -C "$q38_repo_dir" rev-parse HEAD)
        if [[ -n $(git -C "$q38_repo_dir" status --porcelain) ]]; then q38_revision+=-dirty; fi
        exec docker build --network=host \
            --build-arg "SGLANG_SOURCE_REVISION=$q38_revision" \
            --file "$q38_repo_dir/docker/v100-qwen38-flash-next-v5.Dockerfile" \
            --tag "$q38_image" "$q38_repo_dir"
        ;;
esac
