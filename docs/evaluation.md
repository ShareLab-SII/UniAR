# Evaluation

UniAR evaluation follows a three-step pipeline:

```
1. Batch generate    →  inference/generate_batch.py (multi-GPU)
2. Convert format    →  eval/convert_structure/*.py
3. Score             →  official benchmark tools (external)
```

---

## Step 1: Batch Generation

`inference/generate_batch.py` generates images from a JSONL prompt file across multiple GPUs via `accelerate`, with batched inference and automatic resume.

### Available prompt files

We provide ready-to-use prompt files under [`eval/prompts/`](../eval/prompts/):

| File | Benchmark | Prompts |
|------|-----------|---------|
| `eval/prompts/geneval_long.jsonl` | GenEval (long prompts, default) | 553 |
| `eval/prompts/geneval.jsonl` | GenEval (short prompts) | 553 |
| `eval/prompts/oneig.jsonl` | OneIG-Bench (Text Rendering) | 200 |
| `eval/prompts/longtext.jsonl` | LongText-Bench | 160 |
| `eval/prompts/imgedit.jsonl` | ImgEdit | 737 |

### Per-benchmark inference

Each benchmark has a convenience script under [`scripts/infer/`](../scripts/infer/) with the recommended settings. You can also run `generate_batch.py` directly.

#### GenEval (instruction following)

```bash
# Convenience script
MODEL_PATH=checkpoints/UniAR-RL bash scripts/infer/run_geneval.sh

or

accelerate launch --num_processes 8 inference/generate_batch.py \
    --model_path checkpoints/UniAR-RL \
    --data_path eval/prompts/geneval_long.jsonl \
    --output_path eval/runs \
    --run_name geneval \
    --ar_height 512 --ar_width 512 \
    --upsampling_ratio 2.0 \
    --samples_per_prompt 4 \
    --temperature 0.1 --cfg 2.5
```

#### OneIG-Bench (text rendering)

```bash
MODEL_PATH=checkpoints/UniAR-RL bash scripts/infer/run_oneig.sh

or

accelerate launch --num_processes 8 inference/generate_batch.py \
    --model_path checkpoints/UniAR-RL \
    --data_path eval/prompts/oneig.jsonl \
    --output_path eval/runs \
    --run_name oneig \
    --ar_height 704 --ar_width 1280 \
    --upsampling_ratio 1.0 \
    --samples_per_prompt 4 \
    --temperature 0.1 --cfg 2.0
```

#### LongText-Bench (long text rendering)

```bash
MODEL_PATH=checkpoints/UniAR-RL bash scripts/infer/run_longtext.sh

or

accelerate launch --num_processes 8 inference/generate_batch.py \
    --model_path checkpoints/UniAR-RL \
    --data_path eval/prompts/longtext.jsonl \
    --output_path eval/runs \
    --run_name longtext \
    --ar_height 704 --ar_width 1280 \
    --upsampling_ratio 1.0 \
    --samples_per_prompt 4 \
    --temperature 0.1 --cfg 2.0
```

#### ImgEdit (image editing)

```bash
MODEL_PATH=checkpoints/UniAR-RL bash scripts/infer/run_imgedit.sh

or

accelerate launch --num_processes 8 inference/generate_batch.py \
    --model_path checkpoints/UniAR-RL \
    --data_path eval/prompts/imgedit.jsonl \
    --output_path eval/runs \
    --run_name imgedit \
    --ar_height 512 --ar_width 512 \
    --upsampling_ratio 2.0 \
    --samples_per_prompt 1 \
    --temperature 0.1 --cfg 2.0 \
    --image_root eval/prompts/imgedit_images \
    --image_key input_image
```

### Input format

Each line in the JSONL file:

```json
{"idx": 0, "prompt": "A photo of a red car", "meta": {"tag": "single_object"}}
```

- `idx` — unique integer index (auto-assigned if missing)
- `prompt` — text prompt for generation
- `meta` — optional metadata, passed through to output

### Output structure

```
<output_path>/<run_name>/
├── 00000/
│   ├── metadata.json          # passthrough of the input record
│   └── samples/
│       ├── 0000.png
│       ├── 0001.png
│       └── ...
├── 00001/
│   └── ...
```

Re-running the same command **automatically skips** prompts whose PNGs already exist, making it safe to restart interrupted runs.

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | *(required)* | Path to a UniAR checkpoint |
| `--data_path` | *(required)* | Input JSONL file |
| `--output_path` | `eval/runs` | Root output directory |
| `--run_name` | *(required)* | Subdirectory name under `output_path` |
| `--ar_height` | 512 | AR rollout height |
| `--ar_width` | 512 | AR rollout width |
| `--upsampling_ratio` | 2.0 | Decoder upsampling ratio (1.0–2.0) |
| `--samples_per_prompt` | 4 | Number of images per prompt |
| `--batch_size` | 8 | Per-GPU batch size |
| `--temperature` | 0.1 | Sampling temperature |
| `--cfg` | 2.0 | AR classifier-free guidance scale |
| `--decoder_num_inference_steps` | 28 | SD3 denoising steps |
| `--decoder_cfg_scale` | 1.5 | SD3 decoder CFG scale |
| `--seed` | 0 | Random seed |
| `--attn` | `flash_attention_2` | Attention backend |
| `--image_root` | `None` | Root directory for input images (edit mode) |
| `--image_key` | `None` | Key in `meta` pointing to the relative image path (edit mode) |

---

## Step 2: Format Conversion

Each benchmark expects a different output layout. We provide conversion scripts under [`eval/convert_structure/`](../eval/convert_structure/) to transform the unified output into each benchmark's format.

#### GenEval

Writes per-prompt `metadata.jsonl` with `tag`/`include`/`exclude` fields for the upstream evaluator.

```bash
python eval/convert_structure/geneval.py --run_dir eval/runs/geneval
```

#### OneIG-Bench

Packs 4 samples into a 2x2 grid `.webp`, organized by category and model name.

```bash
python eval/convert_structure/oneig.py \
    --run_dir eval/runs/oneig \
    --images_dir eval/runs/oneig/images \
    --model_name uniar
```

#### LongText-Bench

Flattens samples into `<prompt_id>_<k>.png` flat directory.

```bash
python eval/convert_structure/longtext.py \
    --run_dir eval/runs/longtext \
    --samples_dir eval/runs/longtext/samples_flat
```

#### ImgEdit

Flattens samples into `<original_id>.png` flat directory (one sample per prompt).

```bash
python eval/convert_structure/imgedit.py \
    --run_dir eval/runs/imgedit \
    --samples_dir eval/runs/imgedit/samples_flat
```

---

## Step 3: Scoring

After format conversion, use each benchmark's official evaluation tools. These may require **separate conda environments** due to conflicting dependencies.

| Benchmark | Official repo | Notes |
|-----------|--------------|-------|
| GenEval | [djghosh13/geneval](https://github.com/djghosh13/geneval) | Requires mmdet 2.x (Python 3.9, separate env) |
| OneIG-Bench | [OneIG-Bench/OneIG-Benchmark](https://github.com/OneIG-Bench/OneIG-Benchmark) | Requires Qwen2.5-VL as OCR judge |
| LongText-Bench | [X-Omni-Team/X-Omni](https://github.com/X-Omni-Team/X-Omni) (`textbench/`) | Same env as OneIG |
| ImgEdit | [PKU-YuanGroup/ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit) | Requires GPT-4o-compatible API key |

Refer to each benchmark's official README for installation and scoring instructions.
