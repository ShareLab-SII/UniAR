#!/usr/bin/env bash
# UniAR Multi-Node RL: Training Node
#
# Waits for all reward servers to be ready (via shared directory), then
# launches multi-node accelerate training. Run this on EACH training node
# with a different NODE_RANK.
#
# Required env vars:
#   SHARED_DIR              shared filesystem path for service discovery
#   MODEL_PATH              UniAR checkpoint to RL-finetune
#   DATA_PATH               training data (.jsonl or .yaml)
#   MASTER_ADDR             IP of training node 0
#
# Optional — multi-node config:
#   NNODES                  number of training nodes (default: 2)
#   NODE_RANK               this node's rank, 0-indexed (default: 0)
#   NPROC_PER_NODE          GPUs per node (default: 8)
#   MASTER_PORT             coordination port (default: 29500)
#   WAIT_SERVICES           space-separated services to wait for
#                           (default: "decoder unified_reward ocr")
#
# Optional — training hyperparams:
#   IMAGE_WIDTH/IMAGE_HEIGHT    image resolution (default: 512)
#   TEMPERATURE                 sampling temperature (default: 0.1)
#   LR                         learning rate (default: 5e-6)
#   PER_DEVICE_BS              per-device batch size (default: 2)
#   GRAD_ACC                   gradient accumulation steps (default: 16)
#   NUM_GENERATIONS            GRPO group size (default: 16)
#   MAX_STEPS                  max training steps (default: 1000)
#   SAVE_STEPS                 checkpoint interval (default: 50)
#   REWARD_WEIGHTS             JSON list (default: [1.0, 1.0, 1.0, 1.0])
#   REWARD_FUNCTION_NAMES      Python list (default: [unified_reward, geneval_reward, ocr_reward, hpsv2_reward])
#
# Usage:
#   # Node 0 (master):
#   SHARED_DIR=/shared/run01 MODEL_PATH=... DATA_PATH=... \
#     MASTER_ADDR=<this_ip> NODE_RANK=0 bash scripts/train/run_train_node.sh
#
#   # Node 1:
#   SHARED_DIR=/shared/run01 MODEL_PATH=... DATA_PATH=... \
#     MASTER_ADDR=<node0_ip> NODE_RANK=1 bash scripts/train/run_train_node.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/train/rl:${REPO_ROOT}/train/rl/trl:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

SHARED_DIR="${SHARED_DIR:?Set SHARED_DIR to a shared filesystem path}"
mkdir -p "${SHARED_DIR}"
SHARED_DIR="$(cd "${SHARED_DIR}" && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to your UniAR checkpoint}"
REF_MODEL_PATH="${REF_MODEL_PATH:-}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to your training data}"
MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the IP of training node 0}"

NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

IMAGE_WIDTH="${IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
LR="${LR:-5e-6}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACC="${GRAD_ACC:-16}"
MAX_STEPS="${MAX_STEPS:-500}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
ROLLOUT_SAVE_FREQUENCY="${ROLLOUT_SAVE_FREQUENCY:-1}"
ROLLOUT_MAX_IMAGES="${ROLLOUT_MAX_IMAGES:-64}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-[1.0, 1.0, 1.0, 1.0]}"
REWARD_FUNCTION_NAMES="${REWARD_FUNCTION_NAMES:-[unified_reward, geneval_reward, ocr_reward, hpsv2_reward]}"
DECODE_NUM_INFERENCE_STEPS="${DECODE_NUM_INFERENCE_STEPS:-14}"
DECODE_CFG_SCALE="${DECODE_CFG_SCALE:-1.0}"
BETA="${BETA:-0.01}"
LOSS_TYPE="${LOSS_TYPE:-grpo}"
LOSS_USE_FLOAT="${LOSS_USE_FLOAT:-true}"
BSQ_LOSS_NORMALIZE="${BSQ_LOSS_NORMALIZE:-false}"
LOGITS_USE_TEMPERATURE="${LOGITS_USE_TEMPERATURE:-true}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/train/rl/runs}"
RUN_NAME="${RUN_NAME:-rl_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
ROLLOUT_SAVE_DIR="${ROLLOUT_SAVE_DIR:-${OUTPUT_DIR}/rollouts}"
mkdir -p "${OUTPUT_DIR}"

# ---- Wait for reward servers ----
# Space-separated list of services to wait for before training starts.
WAIT_SERVICES="${WAIT_SERVICES:-decoder unified_reward ocr}"

echo "Training node ${NODE_RANK}/${NNODES} | Waiting for services: ${WAIT_SERVICES}"
MAX_WAIT=3600
elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
    all_ready=true
    not_ready=""
    for svc in $WAIT_SERVICES; do
        if [ ! -f "${SHARED_DIR}/${svc}_ready.flag" ] || [ ! -f "${SHARED_DIR}/${svc}_config.json" ]; then
            all_ready=false
            not_ready="${not_ready} ${svc}"
        fi
    done
    if [ "$all_ready" = true ]; then
        echo "All services ready."
        break
    fi
    echo "  waiting... (${elapsed}s, not ready:${not_ready})"
    sleep 30
    elapsed=$((elapsed + 30))
done
if [ "$all_ready" != true ]; then
    echo "ERROR: timeout waiting for services after ${MAX_WAIT}s"
    exit 1
fi

# ---- Read service addresses ----
if [[ " ${WAIT_SERVICES} " == *" decoder "* ]]; then
    DECODER_GPU_URLS_JSON="${SHARED_DIR}/decoder_config.json"
fi
if [[ " ${WAIT_SERVICES} " == *" unified_reward "* ]]; then
    export UNIFIED_REWARD_API_ADDRESS=$(cat "${SHARED_DIR}/unified_reward_config.json" | tr -d '"' | tr -d '\n')
fi
if [[ " ${WAIT_SERVICES} " == *" ocr "* ]]; then
    export OCR_API_ADDRESS=$(cat "${SHARED_DIR}/ocr_config.json" | tr -d '"' | tr -d '\n')
fi
if [[ " ${WAIT_SERVICES} " == *" geneval "* ]]; then
    export GENEVAL_API_ADDRESS=$(cat "${SHARED_DIR}/geneval_config.json" | tr -d '"' | tr -d '\n')
fi
if [[ " ${WAIT_SERVICES} " == *" hpsv2 "* ]]; then
    export HPSV2_API_ADDRESS=$(cat "${SHARED_DIR}/hpsv2_config.json" | tr -d '"' | tr -d '\n')
fi

NUM_PROCESSES=$((NNODES * NPROC_PER_NODE))

echo "============================================"
echo "UniAR GRPO RL Training"
echo "============================================"
echo "Model:    ${MODEL_PATH}"
[ -n "${REF_MODEL_PATH}" ] && echo "Ref:      ${REF_MODEL_PATH}"
echo "Data:     ${DATA_PATH}"
echo "Output:   ${OUTPUT_DIR}"
echo "Training: ${NNODES} nodes x ${NPROC_PER_NODE} GPUs = ${NUM_PROCESSES} processes"
echo "Node:     rank=${NODE_RANK}, master=${MASTER_ADDR}:${MASTER_PORT}"
echo "Services:"
[ -n "${DECODER_GPU_URLS_JSON:-}" ]     && echo "  decode:         $(cat ${DECODER_GPU_URLS_JSON})"
[ -n "${UNIFIED_REWARD_API_ADDRESS:-}" ] && echo "  unified_reward: ${UNIFIED_REWARD_API_ADDRESS}"
[ -n "${OCR_API_ADDRESS:-}" ]            && echo "  ocr:            ${OCR_API_ADDRESS}"
[ -n "${GENEVAL_API_ADDRESS:-}" ]        && echo "  geneval:        ${GENEVAL_API_ADDRESS}"
[ -n "${HPSV2_API_ADDRESS:-}" ]          && echo "  hpsv2:          ${HPSV2_API_ADDRESS}"
echo "============================================"

export WANDB_PROJECT="${WANDB_PROJECT:-uniar_grpo}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_NAME="${RUN_NAME}"
export WANDB_DISABLE_CODE=true
export WANDB_CONSOLE=off
export WANDB_LOG_MODEL=false
export WANDB_WATCH=false
export WANDB__DISABLE_STATS=true

ACCELERATE_CONFIG="${REPO_ROOT}/scripts/train/accelerate_configs/deepspeed_zero2.yaml"

EXTRA_ARGS=()
if [ -n "${REF_MODEL_PATH}" ]; then
    EXTRA_ARGS+=(--ref_model_name_or_path "${REF_MODEL_PATH}")
fi

accelerate launch \
    --config_file="${ACCELERATE_CONFIG}" \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines "${NNODES}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --machine_rank "${NODE_RANK}" \
    train/rl/train_grpo.py \
        --model_name_or_path "${MODEL_PATH}" \
        --attn_implementation flash_attention_2 \
        --data_path "${DATA_PATH}" \
        --image_width "${IMAGE_WIDTH}" \
        --image_height "${IMAGE_HEIGHT}" \
        --temperature "${TEMPERATURE}" \
        --num_generations "${NUM_GENERATIONS}" \
        --max_steps "${MAX_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --decoder_gpu_urls "${DECODER_GPU_URLS_JSON:?decoder must be in WAIT_SERVICES}" \
        --output_dir "${OUTPUT_DIR}" \
        --learning_rate "${LR}" \
        --lr_scheduler_type linear \
        --gradient_accumulation_steps "${GRAD_ACC}" \
        --per_device_train_batch_size "${PER_DEVICE_BS}" \
        --reward_weights_str "${REWARD_WEIGHTS}" \
        --reward_function_names_str "${REWARD_FUNCTION_NAMES}" \
        --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT:-False}" \
        --run_name "${RUN_NAME}" \
        --logging_steps 1 \
        --max_prompt_length 1024 \
        --rollout_save_dir "${ROLLOUT_SAVE_DIR}" \
        --rollout_save_frequency "${ROLLOUT_SAVE_FREQUENCY}" \
        --rollout_max_images_per_step "${ROLLOUT_MAX_IMAGES}" \
        --decode_num_inference_steps "${DECODE_NUM_INFERENCE_STEPS}" \
        --decode_cfg_scale "${DECODE_CFG_SCALE}" \
        --beta "${BETA}" \
        --loss_type "${LOSS_TYPE}" \
        --loss_use_float "${LOSS_USE_FLOAT}" \
        --bsq_loss_normalize "${BSQ_LOSS_NORMALIZE}" \
        --logits_use_temperature "${LOGITS_USE_TEMPERATURE}" \
        "${EXTRA_ARGS[@]}"

echo "Training complete (node ${NODE_RANK})."
