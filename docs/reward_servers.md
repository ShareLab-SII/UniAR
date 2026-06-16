# Reward Server Environments

Each reward server may need its **own conda environment** due to conflicting dependencies. Only install the servers you plan to use — configure `REWARD_FUNCTION_NAMES` in the training script to match.

---

## decode (BSQ visual codes → image)

Uses the same `uniar` environment — no extra setup needed.

```bash
conda activate uniar

SD3_TRANSFORMER_PATH=/path/to/sd3_transformer \
SD3_PATH=/path/to/sd3_pipeline \
IMAGE_TOKENIZER_PATH=/path/to/bsq_encoder \
bash scripts/train/run_reward_server.sh decode
```

| Env var | Description |
|---------|-------------|
| `SD3_TRANSFORMER_PATH` | SD3 transformer checkpoint |
| `SD3_PATH` | SD3 pipeline directory (VAE + text encoders) |
| `IMAGE_TOKENIZER_PATH` | BSQ encoder checkpoint |
| `NUM_GPUS` | Number of GPU workers (default: 1) |
| `DECODE_PORT` | Base port (default: 10000, per-GPU: 10000+N) |

---

## hpsv2 (human preference score)

```bash
conda create -n hpsv2 python=3.10 -y
conda activate hpsv2

pip install open_clip_torch flask gunicorn requests pillow numpy torch

# HPSv2 package
git clone https://github.com/tgxs002/HPSv2 /path/to/HPSv2
export PYTHONPATH=/path/to/HPSv2:$PYTHONPATH
```

**Weights:**

```bash
# HPS v2.1 checkpoint (~40 MB)
# Download from https://github.com/tgxs002/HPSv2#hps-v21

# CLIP ViT-H/14
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
    --local-dir /path/to/CLIP-ViT-H-14
```

**Launch:**

```bash
conda activate hpsv2
HPSV2_CKPT=/path/to/HPS_v2.1_compressed.pt \
CLIP_PATH=/path/to/CLIP-ViT-H-14 \
bash scripts/train/run_reward_server.sh hpsv2
```

---

## geneval (instruction-following score)

Requires **mmdet 2.x** which only installs cleanly on Python 3.9 + Torch 1.12.

```bash
conda create -n geneval python=3.9 -y
conda activate geneval

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

pip install gunicorn flask open-clip-torch numpy opencv-python openmim
mim install mmcv-full mmengine

# mmdet 2.x (from source)
git clone https://github.com/open-mmlab/mmdetection.git /tmp/mmdetection
cd /tmp/mmdetection && git checkout 2.x && pip install -v -e .
```

**Weights:**

```bash
# Mask2Former Swin-S (~440 MB)
wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/\
mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/\
mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth \
    -O /path/to/mask2former.pth
```

**Launch:**

```bash
conda activate geneval
GENEVAL_CONFIG_PATH=/path/to/mask2former_config.py \
GENEVAL_CKPT_PATH=/path/to/mask2former.pth \
bash scripts/train/run_reward_server.sh geneval
```

---

## ocr (text rendering score)

```bash
conda create -n ocr python=3.10 -y
conda activate ocr

pip install paddleocr paddlepaddle-gpu python-Levenshtein flask gunicorn pillow numpy
```

PaddleOCR downloads detection/recognition weights automatically on first use.

**Launch:**

```bash
conda activate ocr
bash scripts/train/run_reward_server.sh ocr
```

---

## unified_reward (VLM judge)

Requires [vLLM](https://github.com/vllm-project/vllm) with Qwen3-VL support. Needs 4+ GPUs for the 32B judge model.

```bash
conda create -n unified_reward python=3.10 -y
conda activate unified_reward

pip install "vllm>=0.9" transformers
```

**Weights:**

```bash
huggingface-cli download CodeGoat24/UnifiedReward-2.0-qwen3vl-32b \
    --local-dir /path/to/UnifiedReward-2.0-qwen3vl-32b
```

**Launch:**

```bash
conda activate unified_reward
MODEL_PATH=/path/to/UnifiedReward-2.0-qwen3vl-32b \
UNIFIED_REWARD_TP=4 \
bash scripts/train/run_reward_server.sh unified_reward
```

| Env var | Description |
|---------|-------------|
| `MODEL_PATH` | HF repo or local path of the judge model |
| `UNIFIED_REWARD_TP` | Tensor parallelism (default: 4) |
| `UNIFIED_REWARD_DP` | Data parallelism (default: 1) |
| `UNIFIED_PORT` | Server port (default: 10010) |
