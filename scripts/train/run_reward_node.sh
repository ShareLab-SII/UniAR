#!/usr/bin/env bash
# UniAR Multi-Node RL: Reward Server Node
#
# Launches selected reward servers on a single node. Writes service URLs to
# the shared directory for training node discovery.
#
# Usage:
#   bash scripts/train/run_reward_node.sh <reward1> [reward2] ...
#
# Available rewards:
#   unified_reward   VLM judge via vLLM (GPU, requires UNIFIED_REWARD_MODEL)
#   geneval          object detection reward (GPU, requires GENEVAL_CONFIG_PATH + GENEVAL_CKPT_PATH)
#   hpsv2            aesthetic score (GPU, requires HPSV2_CKPT + CLIP_PATH)
#   ocr              OCR accuracy reward (CPU, requires nothing or OCR_MODEL_DIR)
#
# Examples:
#   # Only unified_reward and ocr:
#   bash scripts/train/run_reward_node.sh unified_reward ocr
#
#   # All four rewards:
#   bash scripts/train/run_reward_node.sh unified_reward geneval hpsv2 ocr
#
# Required env vars:
#   SHARED_DIR              shared filesystem path for service discovery
#
# Per-reward env vars (see each section below for details):
#   UNIFIED_REWARD_MODEL, UNIFIED_REWARD_GPUS, UNIFIED_REWARD_PORT, UNIFIED_REWARD_ENV
#   GENEVAL_CONFIG_PATH, GENEVAL_CKPT_PATH, GENEVAL_GPUS, GENEVAL_PORT, GENEVAL_ENV
#   HPSV2_CKPT, CLIP_PATH, HPSV2_GPUS, HPSV2_PORT, HPSV2_ENV
#   OCR_MODEL_DIR, OCR_PORT, OCR_ENV
#
# Env activation (per-reward *_ENV):
#   Pass a conda env name (e.g. "geneval") or a venv path (e.g. "/opt/venvs/ocr").
#   Paths containing "/" are treated as venv; plain names as conda.
#   Set CONDA_ROOT if conda is not at ~/miniconda3.

set -euo pipefail

REWARDS=("$@")
if [ ${#REWARDS[@]} -eq 0 ]; then
    echo "Usage: $0 <reward1> [reward2] ..."
    echo "Available: unified_reward, geneval, hpsv2, ocr"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/train/rl:${PYTHONPATH:-}"

SHARED_DIR="${SHARED_DIR:?Set SHARED_DIR to a shared filesystem path}"
mkdir -p "${SHARED_DIR}"
SHARED_DIR="$(cd "${SHARED_DIR}" && pwd)"

LOCAL_IP=$(hostname -I 2>/dev/null | awk '{for(i=1;i<=NF;i++){if($i !~ /^127\./ && $i !~ /^172\.17\./ && $i !~ /^172\.18\./){print $i; exit}}}')
[ -z "$LOCAL_IP" ] && LOCAL_IP=$(ip route get 8.8.8.8 2>/dev/null | awk '{print $7; exit}' | head -1)

write_service_info() {
    local name="$1" url="$2"
    echo "${url}" > "${SHARED_DIR}/${name}_config.json"
    date '+%Y-%m-%d %H:%M:%S' > "${SHARED_DIR}/${name}_ready.flag"
    echo "  [${name}] ready: ${url}"
}

wait_for_health() {
    local name="$1" url="$2" max_wait="${3:-600}" elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        curl -s "${url}" > /dev/null 2>&1 && return 0
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "ERROR: ${name} health check timeout (${max_wait}s)"
    return 1
}

CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"

activate_env() {
    local env="$1"
    if [ -z "${env}" ]; then
        return
    fi
    if [[ "${env}" == */* ]]; then
        # Path-like: treat as venv
        source "${env}/bin/activate"
        echo "  activated venv: ${env}"
    else
        # Name-like: treat as conda env
        # shellcheck disable=SC1091
        source "${CONDA_ROOT}/etc/profile.d/conda.sh"
        conda activate "${env}"
        echo "  activated conda env: ${env}"
    fi
}

echo "Reward node | IP: ${LOCAL_IP} | Rewards: ${REWARDS[*]}"

PIDS=()

for reward in "${REWARDS[@]}"; do
case "${reward}" in

    unified_reward)
        UNIFIED_REWARD_MODEL="${UNIFIED_REWARD_MODEL:?Set UNIFIED_REWARD_MODEL}"
        UNIFIED_REWARD_PORT="${UNIFIED_REWARD_PORT:-10010}"
        UNIFIED_REWARD_GPUS="${UNIFIED_REWARD_GPUS:-0,1,2,3}"
        IFS=',' read -ra _UR_GPUS <<< "${UNIFIED_REWARD_GPUS}"
        UNIFIED_REWARD_TP="${#_UR_GPUS[@]}"

        echo "Starting unified_reward on GPUs ${UNIFIED_REWARD_GPUS} (TP=${UNIFIED_REWARD_TP})..."
        (
            activate_env "${UNIFIED_REWARD_ENV:-}"
            CUDA_VISIBLE_DEVICES=${UNIFIED_REWARD_GPUS} \
            MODEL_PATH="${UNIFIED_REWARD_MODEL}" \
            PORT="${UNIFIED_REWARD_PORT}" \
            TENSOR_PARALLEL="${UNIFIED_REWARD_TP}" \
            CONFIG_FILE="${SHARED_DIR}/unified_reward_config.json" \
            READY_MARKER_FILE="${SHARED_DIR}/unified_reward_ready.flag" \
            LOG_FILE="${SHARED_DIR}/unified_reward.log" \
            bash train/rl/reward_server/unified_reward/vllm_server.sh
        ) &
        PIDS+=($!)
        ;;

    geneval)
        GENEVAL_CONFIG_PATH="${GENEVAL_CONFIG_PATH:?Set GENEVAL_CONFIG_PATH}"
        GENEVAL_CKPT_PATH="${GENEVAL_CKPT_PATH:?Set GENEVAL_CKPT_PATH}"
        GENEVAL_PORT="${GENEVAL_PORT:-10011}"
        GENEVAL_GPUS="${GENEVAL_GPUS:-4,5}"
        IFS=',' read -ra _GE_GPUS <<< "${GENEVAL_GPUS}"
        GENEVAL_NUM_DEVICES="${#_GE_GPUS[@]}"

        echo "Starting geneval on GPUs ${GENEVAL_GPUS} (${GENEVAL_NUM_DEVICES} workers)..."
        (
            activate_env "${GENEVAL_ENV:-}"
            cd train/rl/reward_server/geneval
            export GENEVAL_CONFIG_PATH GENEVAL_CKPT_PATH
            GPU_IDS="[${GENEVAL_GPUS}]" NUM_DEVICES="${GENEVAL_NUM_DEVICES}" PORT="${GENEVAL_PORT}" \
                gunicorn -c gunicorn.conf.py server:app &
            SRV_PID=$!
            wait_for_health "geneval" "http://${LOCAL_IP}:${GENEVAL_PORT}/health"
            write_service_info "geneval" "http://${LOCAL_IP}:${GENEVAL_PORT}"
            wait $SRV_PID
        ) &
        PIDS+=($!)
        ;;

    hpsv2)
        HPSV2_CKPT="${HPSV2_CKPT:?Set HPSV2_CKPT}"
        CLIP_PATH="${CLIP_PATH:?Set CLIP_PATH}"
        HPSV2_PORT="${HPSV2_PORT:-10012}"
        HPSV2_GPUS="${HPSV2_GPUS:-6,7}"
        IFS=',' read -ra _HP_GPUS <<< "${HPSV2_GPUS}"
        HPSV2_NUM_DEVICES="${#_HP_GPUS[@]}"

        echo "Starting hpsv2 on GPUs ${HPSV2_GPUS} (${HPSV2_NUM_DEVICES} workers)..."
        (
            activate_env "${HPSV2_ENV:-}"
            cd train/rl/reward_server/hpsv2
            export HPSV2_CKPT CLIP_PATH
            GPU_IDS="[${HPSV2_GPUS}]" NUM_DEVICES="${HPSV2_NUM_DEVICES}" PORT="${HPSV2_PORT}" \
                gunicorn -c gunicorn.conf.py server:app &
            SRV_PID=$!
            wait_for_health "hpsv2" "http://${LOCAL_IP}:${HPSV2_PORT}/health"
            write_service_info "hpsv2" "http://${LOCAL_IP}:${HPSV2_PORT}"
            wait $SRV_PID
        ) &
        PIDS+=($!)
        ;;

    ocr)
        OCR_PORT="${OCR_PORT:-10013}"
        echo "Starting ocr on port ${OCR_PORT} (CPU)..."
        (
            activate_env "${OCR_ENV:-}"
            cd train/rl/reward_server/ocr
            export OCR_MODEL_DIR="${OCR_MODEL_DIR:-}"
            NUM_DEVICES=1 PORT="${OCR_PORT}" \
                gunicorn -c gunicorn.conf.py server:app &
            SRV_PID=$!
            wait_for_health "ocr" "http://${LOCAL_IP}:${OCR_PORT}/health" 300
            write_service_info "ocr" "http://${LOCAL_IP}:${OCR_PORT}"
            wait $SRV_PID
        ) &
        PIDS+=($!)
        ;;

    *)
        echo "ERROR: unknown reward '${reward}'. Available: unified_reward, geneval, hpsv2, ocr"
        exit 1
        ;;
esac
done

echo "All ${#REWARDS[@]} reward servers launching. Waiting..."
wait "${PIDS[@]}" 2>/dev/null || true
sleep infinity
