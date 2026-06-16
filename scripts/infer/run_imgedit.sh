#!/usr/bin/env bash
# ImgEdit inference: generate edited images from ImgEdit prompts (edit mode).

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the UniAR checkpoint directory}"
NUM_GPUS="${NUM_GPUS:-8}"
DATA_PATH="${DATA_PATH:-eval/prompts/imgedit.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-eval/runs}"
RUN_NAME="${RUN_NAME:-imgedit}"

accelerate launch --num_processes "${NUM_GPUS}" \
    inference/generate_batch.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --run_name "${RUN_NAME}" \
    --ar_height 512 \
    --ar_width 512 \
    --upsampling_ratio 2.0 \
    --samples_per_prompt 1 \
    --batch_size 8 \
    --temperature 0.1 \
    --cfg 2.0 \
    --image_root eval/prompts/imgedit_images \
    --image_key input_image

echo "Done. Results: ${OUTPUT_PATH}/${RUN_NAME}"
