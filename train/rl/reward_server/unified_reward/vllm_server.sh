#!/bin/bash
# Launch a vLLM server for the Unified Reward VLM judge.
#
# Env vars:
#   MODEL_PATH          (required) HF model path, e.g. CodeGoat24/UnifiedReward-2.0-qwen3vl-32b
#   PORT                server port (default: 8080)
#   TENSOR_PARALLEL     tensor parallel size (default: 4)
#   PIPELINE_PARALLEL   pipeline parallel size (default: 1)
#   GPU_MEMORY_UTILIZATION  vLLM GPU memory fraction (default: 0.8)
#   SERVER_NAME         served model name (default: UnifiedReward)
#   LOG_FILE            log output file (default: stdout)
#   CONFIG_FILE         write service URL to this file when ready
#   READY_MARKER_FILE   create this file when the server is ready

set -euo pipefail

LOCAL_IP=$(hostname -I 2>/dev/null | awk '{for(i=1;i<=NF;i++){if($i !~ /^127\./ && $i !~ /^172\.17\./ && $i !~ /^172\.18\./){print $i; exit}}}')
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP=$(ip route get 8.8.8.8 2>/dev/null | awk '{print $7; exit}' | head -1)
fi

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be set}"
PORT="${PORT:-8080}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
PIPELINE_PARALLEL="${PIPELINE_PARALLEL:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
SERVER_NAME="${SERVER_NAME:-UnifiedReward}"
LOG_FILE="${LOG_FILE:-/dev/stdout}"

echo "Starting Unified Reward server..."
echo "  model: ${MODEL_PATH}"
echo "  host:  ${LOCAL_IP}:${PORT}"
echo "  TP=${TENSOR_PARALLEL}, PP=${PIPELINE_PARALLEL}"

vllm serve "${MODEL_PATH}" \
    --host "${LOCAL_IP}" \
    --trust-remote-code \
    --served-model-name "${SERVER_NAME}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --pipeline-parallel-size "${PIPELINE_PARALLEL}" \
    --limit-mm-per-prompt.image 16 \
    --enable-prefix-caching \
    --port "${PORT}" > "${LOG_FILE}" 2>&1 &
VLLM_PID=$!

SERVICE_URL="http://${LOCAL_IP}:${PORT}"
MAX_WAIT=600
ELAPSED=0

echo "Waiting for server to start (PID: ${VLLM_PID})..."
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! ps -p "$VLLM_PID" > /dev/null 2>&1; then
        echo "ERROR: vLLM process exited unexpectedly"
        [ -f "${LOG_FILE}" ] && tail -50 "${LOG_FILE}"
        exit 1
    fi
    if curl -s "${SERVICE_URL}/v1/models" > /dev/null 2>&1; then
        echo "Server is ready at ${SERVICE_URL}"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "ERROR: server startup timed out after ${MAX_WAIT}s"
    [ -f "${LOG_FILE}" ] && tail -50 "${LOG_FILE}"
    exit 1
fi

if [ -n "${CONFIG_FILE:-}" ]; then
    mkdir -p "$(dirname "$CONFIG_FILE")"
    echo "${SERVICE_URL}" > "$CONFIG_FILE"
fi

if [ -n "${READY_MARKER_FILE:-}" ]; then
    mkdir -p "$(dirname "$READY_MARKER_FILE")"
    date '+%Y-%m-%d %H:%M:%S' > "$READY_MARKER_FILE"
fi

disown $VLLM_PID 2>/dev/null || true
echo "Server running in background (PID: ${VLLM_PID}). Stop with: kill ${VLLM_PID}"
