# Inference

UniAR provides three inference entry points:

| Script | Purpose | Typical use |
|--------|---------|-------------|
| [`inference/chat.py`](#image-understanding) | Image understanding (VQA) | Quick demo, single image + question |
| [`inference/generate.py`](#image-generation) | Single-prompt text-to-image | Quick demo, one prompt → one image |
| [`inference/generate_batch.py`](#batch-generation) | Batched multi-GPU generation | Benchmark evaluation, large-scale runs |

---

## Image Understanding

`inference/chat.py` takes one image and one text prompt, generates a text response.

```bash
python inference/chat.py \
    --model_path checkpoints/UniAR-RL \
    --image https://example.com/photo.jpg \
    --prompt "Describe this image in detail."
```

`--image` accepts a URL or a local file path.

**Parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | *(required)* | Path to a UniAR checkpoint |
| `--image` | *(required)* | Input image (URL or local path) |
| `--prompt` | *(required)* | Text question about the image |
| `--max_new_tokens` | 1024 | Maximum tokens to generate |
| `--attn` | `flash_attention_2` | Attention backend (`flash_attention_2` / `sdpa` / `eager`) |

---

## Image Generation

`inference/generate.py` generates one image from one text prompt.

```bash
python inference/generate.py \
    --model_path checkpoints/UniAR-RL \
    --prompt "A cinematic photo of a corgi wearing sunglasses." \
    --output_path output.png
```

**Parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | *(required)* | Path to a UniAR checkpoint |
| `--prompt` | *(required)* | Text prompt for image generation |
| `--output_path` | `inference/generated.png` | Output PNG path |
| `--ar_height` | 960 | AR rollout height (pixels) |
| `--ar_width` | 960 | AR rollout width (pixels) |
| `--upsampling_ratio` | 1.067 | Decoder output scale relative to AR resolution (960*1.067=1024) |
| `--temperature` | 1.0 | Sampling temperature for visual tokens |
| `--cfg` | 1.5 | Classifier-free guidance scale for AR rollout |
| `--decoder_num_inference_steps` | 28 | SD3 visual decoder denoising steps |
| `--decoder_cfg_scale` | 1.5 | SD3 visual decoder decoder CFG scale |
| `--attn` | `flash_attention_2` | Attention backend |

**Prompt tips:**

UniAR supports rendering text in generated images. To do so, enclose the target text in double quotes within your prompt:

```bash
python inference/generate.py \
    --model_path checkpoints/UniAR-RL \
    --prompt 'A cute little kitten wearing a sign around its neck that says \"UniAR\", adorable expression, soft fluffy fur, big sparkling eyes, charming and playful, high detail, warm lighting, clean background, wholesome and visually appealing, digital illustration style.' \
    --output_path inference/generated.png
```

For better image quality, we recommend **rewriting short prompts** into detailed descriptions using an LLM (e.g. Qwen). Adding details about style, lighting, composition, and resolution can significantly improve generation results. You can use the prompt optimizer from [Qwen-Image](https://github.com/QwenLM/Qwen-Image#prompt-enhance-for-text-to-image).

**Resolution notes:**

The final output resolution is `ar_height * upsampling_ratio` x `ar_width * upsampling_ratio`. The visual decoder supports upsampling ratios from **1x to 2x**. Recommended AR resolutions:

| `ar_height` x `ar_width` | `upsampling_ratio` | Output resolution |
|---------------------------|-------------------|-------------------|
| 512 x 512 | 2.0 | 1024 x 1024 |
| 960 x 960 | 1.067 | 1024 x 1024 |
| 1280 x 704 | 1.0 | 1280 x 704 |
| 704 x 1280 | 1.0 | 704 x 1280 |

---

## Batch Generation

For benchmark evaluation and large-scale generation, use `inference/generate_batch.py` which supports **multi-GPU** distributed generation via `accelerate`, batched inference, and automatic resume.

See [docs/evaluation.md](evaluation.md) for the full usage guide, available prompt files, output format, and edit mode.
