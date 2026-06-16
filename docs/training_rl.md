# RL Training (GRPO)

UniAR uses **GRPO** (Group Relative Policy Optimization) with a multi-reward stack to improve image generation quality via reinforcement learning.

## Architecture

The RL training system consists of three types of nodes that communicate via HTTP and a shared filesystem:

```
┌─── Decode Node ──────────┐   ┌─── Reward Node ──────────────────┐
│  decode server           │   │  hpsv2                           │
│  (BSQ visual codes → PNG)│   │  geneval                         │
└──────────┬───────────────┘   │  ocr                             │
           │                   │  unified_reward                  │
           │                   └──────────┬───────────────────────┘
           │  HTTP                        │  HTTP
           ▼                              ▼
┌─── Training Node(s) ────────────────────────────────────────────┐
│  train_grpo.py → UniARGRPOTrainer                               │
│    1. AR rollout → BSQ visual codes                             │
│    2. Send codes to decode server → PNG                         │
│    3. Send PNG to reward servers → scores                       │
│    4. GRPO loss update                                          │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
    SHARED_DIR (shared filesystem)
    ├── decoder_config.json     # decode server URLs
    ├── decoder_ready.flag
    ├── unified_reward_config.json
    ├── unified_reward_ready.flag
    ├── ...
```

All nodes discover each other via **flag files** in a shared directory (`SHARED_DIR`). Training nodes wait for all services to be ready before starting. A typical deployment uses 3–4 nodes: 1 decode + 1 reward + 1–2 training (8 GPUs each).

---

## Quick Start (Single-Node)

For a minimal test on a single machine, use `run_reward_server.sh` to start servers individually:

```bash
conda activate uniar

# Terminal 1: decode server (uses UniAR decoder weights)
SD3_TRANSFORMER_PATH=/path/to/sd3_transformer \
SD3_PATH=/path/to/sd3_pipeline \
IMAGE_TOKENIZER_PATH=/path/to/bsq_encoder \
bash scripts/train/run_reward_server.sh decode

# Terminal 2: hpsv2 reward
HPSV2_CKPT=/path/to/HPS_v2.1_compressed.pt \
CLIP_PATH=/path/to/CLIP-ViT-H-14 \
bash scripts/train/run_reward_server.sh hpsv2

# Terminal 3: training (minimal config)
MODEL_PATH=/path/to/ar_model \
DATA_PATH=data/rl_s1/s1_mix.yaml \
MASTER_ADDR=localhost \
SHARED_DIR=/tmp/uniar_rl \
REWARD_WEIGHTS="[1.0]" \
REWARD_FUNCTION_NAMES="[hpsv2_reward]" \
WAIT_SERVICES="decoder" \
MAX_STEPS=10 \
bash scripts/train/run_train_node.sh
```

---

## Multi-Node Setup

For production training, run each service type on a dedicated node.

### Step 1: Start decode servers

```bash
# On the decode node
SHARED_DIR=/shared/run01 \
SD3_TRANSFORMER_PATH=/path/to/sd3_transformer \
SD3_PATH=/path/to/sd3_pipeline \
IMAGE_TOKENIZER_PATH=/path/to/bsq_encoder \
DECODE_NUM_GPUS=8 \
bash scripts/train/run_decode_node.sh
```

### Step 2: Start reward servers

```bash
# On the reward node (launches selected rewards)
SHARED_DIR=/shared/run01 \
UNIFIED_REWARD_MODEL=CodeGoat24/UnifiedReward-2.0-qwen3vl-32b \
GENEVAL_CONFIG_PATH=/path/to/mask2former.py \
GENEVAL_CKPT_PATH=/path/to/mask2former_weights \
HPSV2_CKPT=/path/to/HPS_v2.1_compressed.pt \
CLIP_PATH=/path/to/CLIP-ViT-H-14 \
bash scripts/train/run_reward_node.sh unified_reward geneval hpsv2 ocr
```

### Step 3: Launch training

```bash
# On each training node
SHARED_DIR=/shared/run01 \
MODEL_PATH=/path/to/ar_model \
DATA_PATH=data/rl_s1/s1_mix.yaml \
MASTER_ADDR=<training_node_0_ip> \
NODE_RANK=0 \
bash scripts/train/run_train_node.sh
```

For multi-node training, run the same command on each node with a different `NODE_RANK`.

---

## Training Data Format

Each JSONL record:

```json
{
  "instruction": "A photo of a cat sitting on a windowsill",
  "number": 12345,
  "task": "geneval",
  "metadata": {"tag": "single_object", "include": ["cat"]}
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `instruction` | Yes | Text prompt for image generation |
| `number` | Yes | Unique integer ID (used for logging) |
| `task` | No | Specifies which reward functions to apply. Supported values: `geneval`, `ocr`, or omitted/`null` (general-purpose rewards only). See [`train/rl/reward_funcs.py`](../train/rl/reward_funcs.py) for the reward routing logic |
| `metadata` | No | Reward-specific metadata (e.g. GenEval `tag`/`include`/`exclude`) |

> **Note on OCR prompts:** For `task: "ocr"`, the target text to be rendered must be enclosed in double quotes within the instruction. The OCR reward function extracts quoted text as the ground truth for evaluation. Example: `"A store sign that reads \"Open 24 Hours\" in bold neon letters"`.

**YAML mix config** — sample from multiple JSONL files:

```yaml
datasets:
  - name: blip3o_60k
    json_path: data/rl_s1/blip3o_60k.jsonl
    sampled_ratio: 10000       # integer: exact count
  - name: geneval_raw
    json_path: data/rl_s1/geneval_raw.jsonl
    sampled_ratio: 10000
```

`sampled_ratio`: integer = exact sample count; float in (0,1) = fraction.

---

## Key Training Parameters

Set via environment variables before calling `run_train_node.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | *(required)* | AR model checkpoint |
| `DATA_PATH` | *(required)* | Training data (JSONL or YAML) |
| `MASTER_ADDR` | *(required)* | IP of training node 0 |
| `SHARED_DIR` | *(required)* | Shared filesystem for service discovery |
| `NNODES` | 2 | Number of training nodes |
| `NODE_RANK` | 0 | This node's rank |
| `NPROC_PER_NODE` | 8 | GPUs per node |
| `LR` | 5e-6 | Learning rate |
| `BETA` | 0.01 | GRPO KL penalty coefficient |
| `LOSS_TYPE` | grpo | Loss type (`grpo` or `dapo`) |
| `TEMPERATURE` | 1.0 | Sampling temperature for rollouts |
| `NUM_GENERATIONS` | 16 | GRPO group size |
| `PER_DEVICE_BS` | 2 | Per-device batch size |
| `GRAD_ACC` | 16 | Gradient accumulation steps |
| `MAX_STEPS` | 500 | Maximum training steps |
| `SAVE_STEPS` | 50 | Checkpoint save interval |
| `REWARD_WEIGHTS` | `[1.0, 1.0, 1.0, 1.0]` | Per-reward weights |
| `REWARD_FUNCTION_NAMES` | `[unified_reward, geneval_reward, ocr_reward, hpsv2_reward]` | Active reward functions |
| `WAIT_SERVICES` | `decoder unified_reward ocr` | Services to wait for before starting |

---

## Reward Servers

| Server | Conda env | GPU | Key weights |
|--------|-----------|-----|-------------|
| **decode** | `uniar` (same as training) | 1+ GPU | SD3 transformer + pipeline + BSQ encoder |
| **hpsv2** | `hpsv2` | CPU/GPU | HPSv2 + CLIP-ViT-H-14 |
| **geneval** | `geneval` (Python 3.9) | GPU | Mask2Former (mmdet 2.x) |
| **ocr** | `ocr` | CPU | PaddleOCR (auto-downloaded) |
| **unified_reward** | `unified_reward` | 4+ GPU | UnifiedReward-2.0-qwen3vl-32b via vLLM |

See [docs/reward_servers.md](reward_servers.md) for detailed environment setup, weight downloads, and launch commands for each server.

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/train/run_train_node.sh` | Launch training on one node |
| `scripts/train/run_decode_node.sh` | Start decode servers + register in SHARED_DIR |
| `scripts/train/run_reward_node.sh` | Start multiple reward servers + register in SHARED_DIR |
| `scripts/train/run_reward_server.sh` | Start a single reward server (standalone) |
