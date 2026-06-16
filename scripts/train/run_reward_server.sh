#!/usr/bin/env bash
# Single-reward-server launcher template.
#
# Usage:
#   bash scripts/train/run_reward_server.sh <name>
#
# where <name> ∈ { hpsv2 | geneval | ocr | unified_reward | decode }.
#
# Each branch documents the env vars it consumes; most of them have sensible
# defaults for a single-machine setup. For production you'll typically run
# each server on a separate host/GPU set and point the training driver at it
# via the matching *_API_ADDRESS env var (see train_example.sh).
#
# Port conventions (override via the *_PORT vars below):
#   10000 — decode base port (per-GPU ports are 10000+N)
#   10010 — unified_reward (vLLM)
#   10011 — hpsv2
#   10012 — geneval
#   10013 — ocr
#
# Each reward server has its own conda env (see reward_server/<name>/README.md
# for package requirements). Activate the env BEFORE invoking this script, or
# set CONDA_ENV=<name> and CONDA_ROOT=<path> to have it auto-activate.

set -euo pipefail

NAME="${1:-}"
if [ -z "${NAME}" ]; then
    echo "Usage: $0 <hpsv2|geneval|ocr|unified_reward|decode>"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/train/rl:${PYTHONPATH:-}"

# Optional conda activation.
if [ -n "${CONDA_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_ROOT:-${HOME}/miniconda3}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

case "${NAME}" in
    hpsv2)
        # See reward_server/hpsv2/README.md for the weight download.
        export HPSV2_CKPT="${HPSV2_CKPT:?HPSV2_CKPT must point to HPS_v2.1_compressed.pt}"
        export CLIP_PATH="${CLIP_PATH:?CLIP_PATH must point to the CLIP-ViT-H-14 weights}"
        PORT="${HPSV2_PORT:-10011}"
        NUM_WORKERS="${HPSV2_NUM_WORKERS:-4}"
        cd "${REPO_ROOT}/train/rl/reward_server/hpsv2"
        exec gunicorn -c gunicorn.conf.py \
            --bind "0.0.0.0:${PORT}" --workers "${NUM_WORKERS}" \
            server:app
        ;;

    geneval)
        # See reward_server/geneval/README.md for the mmdetection weights.
        export GENEVAL_CONFIG_PATH="${GENEVAL_CONFIG_PATH:?GENEVAL_CONFIG_PATH must point to mmdetection config .py}"
        export GENEVAL_CKPT_PATH="${GENEVAL_CKPT_PATH:?GENEVAL_CKPT_PATH must point to mmdetection ckpt dir}"
        PORT="${GENEVAL_PORT:-10012}"
        NUM_WORKERS="${GENEVAL_NUM_WORKERS:-2}"
        cd "${REPO_ROOT}/train/rl/reward_server/geneval"
        exec gunicorn -c gunicorn.conf.py \
            --bind "0.0.0.0:${PORT}" --workers "${NUM_WORKERS}" \
            server:app
        ;;

    ocr)
        # PaddleOCR pulls weights on first run unless OCR_MODEL_DIR is set.
        export OCR_MODEL_DIR="${OCR_MODEL_DIR:-}"
        PORT="${OCR_PORT:-10013}"
        NUM_WORKERS="${OCR_NUM_WORKERS:-8}"
        cd "${REPO_ROOT}/train/rl/reward_server/ocr"
        exec gunicorn -c gunicorn.conf.py \
            --bind "0.0.0.0:${PORT}" --workers "${NUM_WORKERS}" \
            server:app
        ;;

    unified_reward)
        # Serves a Qwen3-VL-based judge via vLLM. See reward_server/unified_reward/README.md.
        export MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be the HF repo/path of the judge model (e.g. CodeGoat24/UnifiedReward-2.0-qwen3vl-32b)}"
        export PORT="${UNIFIED_PORT:-10010}"
        export UNIFIED_REWARD_DP="${UNIFIED_REWARD_DP:-1}"
        export UNIFIED_REWARD_TP="${UNIFIED_REWARD_TP:-4}"
        exec bash "${REPO_ROOT}/train/rl/reward_server/unified_reward/vllm_server.sh"
        ;;

    decode)
        # Spawns one decoder process per GPU. See reward_server/decode/README.md
        # for the weight env vars (IMAGE_TOKENIZER_PATH, SD3_TRANSFORMER_PATH, SD3_PATH).
        export SD3_TRANSFORMER_PATH="${SD3_TRANSFORMER_PATH:?SD3_TRANSFORMER_PATH must point to the SD3 transformer checkpoint}"
        export SD3_PATH="${SD3_PATH:?SD3_PATH must point to the SD3 pipeline directory}"
        export IMAGE_TOKENIZER_PATH="${IMAGE_TOKENIZER_PATH:?IMAGE_TOKENIZER_PATH must point to the BSQ encoder checkpoint}"
        PORT="${DECODE_PORT:-10000}"
        NUM_GPUS="${NUM_GPUS:-1}"
        INFERENCE_SKIP_FINAL_LAYERNORM="${INFERENCE_SKIP_FINAL_LAYERNORM:-true}"
        SKIP_NORM_FLAG=""
        if [ "${INFERENCE_SKIP_FINAL_LAYERNORM}" = "true" ]; then
            SKIP_NORM_FLAG="--inference-skip-final-layernorm"
        fi
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            GPU_PORT=$((PORT + i))
            CUDA_VISIBLE_DEVICES=$i python "${REPO_ROOT}/train/rl/reward_server/decode/server.py" \
                --sd3-transformer-path "${SD3_TRANSFORMER_PATH}" \
                --sd3-path "${SD3_PATH}" \
                --image-tokenizer-path "${IMAGE_TOKENIZER_PATH}" \
                --port "${GPU_PORT}" \
                ${SKIP_NORM_FLAG} &
        done
        wait
        ;;

    *)
        echo "Unknown reward server: ${NAME}"
        echo "Supported: hpsv2 | geneval | ocr | unified_reward | decode"
        exit 1
        ;;
esac
