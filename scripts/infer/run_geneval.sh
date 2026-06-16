#!/usr/bin/env bash
# GenEval inference: generate images from GenEval prompts.

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the UniAR checkpoint directory}"
NUM_GPUS="${NUM_GPUS:-8}"
DATA_PATH="${DATA_PATH:-eval/prompts/geneval_long.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-eval/runs}"
RUN_NAME="${RUN_NAME:-geneval}"

accelerate launch --num_processes "${NUM_GPUS}" \
    inference/generate_batch.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --run_name "${RUN_NAME}" \
    --ar_height 512 \
    --ar_width 512 \
    --upsampling_ratio 2.0 \
    --samples_per_prompt 4 \
    --batch_size 8 \
    --temperature 0.1 \
    --cfg 2.5

echo "Done. Results: ${OUTPUT_PATH}/${RUN_NAME}"
