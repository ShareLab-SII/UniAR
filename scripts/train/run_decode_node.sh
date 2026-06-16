#!/usr/bin/env bash
# UniAR Multi-Node RL: Decode Server Node
#
# Launches one BSQ decode server per GPU. Writes the URL list to the shared
# directory so that training nodes can discover it.
#
# Required env vars:
#   SHARED_DIR              shared filesystem path for service discovery
#   SD3_TRANSFORMER_PATH    SD3 transformer checkpoint
#   SD3_PATH                SD3 pipeline directory
#   IMAGE_TOKENIZER_PATH    BSQ encoder checkpoint

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/train/rl:${PYTHONPATH:-}"

SHARED_DIR="${SHARED_DIR:?Set SHARED_DIR to a shared filesystem path}"
SD3_TRANSFORMER_PATH="${SD3_TRANSFORMER_PATH:?Set SD3_TRANSFORMER_PATH}"
SD3_PATH="${SD3_PATH:?Set SD3_PATH}"
IMAGE_TOKENIZER_PATH="${IMAGE_TOKENIZER_PATH:?Set IMAGE_TOKENIZER_PATH}"
DECODE_NUM_GPUS="${DECODE_NUM_GPUS:-8}"
DECODE_BASE_PORT="${DECODE_BASE_PORT:-8000}"
INFERENCE_SKIP_FINAL_LAYERNORM="${INFERENCE_SKIP_FINAL_LAYERNORM:-true}"

mkdir -p "${SHARED_DIR}"
SHARED_DIR="$(cd "${SHARED_DIR}" && pwd)"

LOCAL_IP=$(hostname -I 2>/dev/null | awk '{for(i=1;i<=NF;i++){if($i !~ /^127\./ && $i !~ /^172\.17\./ && $i !~ /^172\.18\./){print $i; exit}}}')
[ -z "$LOCAL_IP" ] && LOCAL_IP=$(ip route get 8.8.8.8 2>/dev/null | awk '{print $7; exit}' | head -1)

echo "Decode node | IP: ${LOCAL_IP} | GPUs: ${DECODE_NUM_GPUS} | Base port: ${DECODE_BASE_PORT}"

# Start one server per GPU
PIDS=()
for i in $(seq 0 $((DECODE_NUM_GPUS - 1))); do
    GPU_PORT=$((DECODE_BASE_PORT + i))
    SKIP_NORM_FLAG=""
    if [ "${INFERENCE_SKIP_FINAL_LAYERNORM}" = "true" ]; then
        SKIP_NORM_FLAG="--inference-skip-final-layernorm"
    fi
    CUDA_VISIBLE_DEVICES=$i python train/rl/reward_server/decode/server.py \
        --sd3-transformer-path "${SD3_TRANSFORMER_PATH}" \
        --sd3-path "${SD3_PATH}" \
        --image-tokenizer-path "${IMAGE_TOKENIZER_PATH}" \
        --port "${GPU_PORT}" \
        ${SKIP_NORM_FLAG} &
    PIDS+=($!)
done

# Wait for all servers to be healthy
URLS="["
for i in $(seq 0 $((DECODE_NUM_GPUS - 1))); do
    GPU_PORT=$((DECODE_BASE_PORT + i))
    URL="http://${LOCAL_IP}:${GPU_PORT}"
    echo "  Waiting for GPU ${i} at ${URL}..."
    elapsed=0
    while [ $elapsed -lt 600 ]; do
        curl -s "${URL}/health" > /dev/null 2>&1 && break
        sleep 5
        elapsed=$((elapsed + 5))
    done
    if [ $elapsed -ge 600 ]; then
        echo "ERROR: GPU ${i} timeout"
        exit 1
    fi
    [ $i -gt 0 ] && URLS="${URLS}, "
    URLS="${URLS}\"${URL}\""
done
URLS="${URLS}]"

# Write config for training nodes
echo "${URLS}" > "${SHARED_DIR}/decoder_config.json"
date '+%Y-%m-%d %H:%M:%S' > "${SHARED_DIR}/decoder_ready.flag"
echo "All ${DECODE_NUM_GPUS} decode servers ready."
echo "Config: ${SHARED_DIR}/decoder_config.json"

wait "${PIDS[@]}"
