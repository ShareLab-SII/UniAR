"""
UniAR batched visual generation launcher (CLI + accelerate data-parallel).

Reads a jsonl file of prompts (unified format: ``{"idx": int, "prompt": str, "meta": dict}``),
runs the AR + SD3 rollout across all visible GPUs, and writes:

    <output_path>/<run_name>/<idx:05d>/metadata.json
    <output_path>/<run_name>/<idx:05d>/samples/<sample_idx:04d>.png

Example::

    accelerate launch --num_processes 8 inference/generate_batch.py \
        --model_path /path/to/UniAR \
        --data_path inference/examples/prompts.jsonl \
        --output_path eval/runs \
        --run_name smoke \
        --samples_per_prompt 4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoProcessor

from uniar import UniARForConditionalGeneration, UniARVisualDecoder

from inference.visual_inputs import prepare_visual_inputs


def _load_prompts(data_path: str) -> List[dict]:
    assert data_path.endswith(".jsonl"), "prompts file must be .jsonl"
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "idx" not in rec:
                rec["idx"] = len(records)
            if "prompt" not in rec:
                raise ValueError(f"record missing 'prompt': {rec}")
            records.append(rec)
    return records


def _build_jobs(
    prompts: List[dict], samples_per_prompt: int, run_dir: Path, resume: bool,
    image_root: Optional[str] = None, image_key: Optional[str] = None,
) -> List[dict]:
    """Expand (prompt) x (sample_idx) into leaf jobs, dropping ones already rendered.

    When ``image_root`` and ``image_key`` are supplied, each job is annotated
    with ``input_image_path`` = ``<image_root>/<meta[image_key]>`` for edit mode.
    """
    jobs = []
    for rec in prompts:
        sample_dir = run_dir / f"{int(rec['idx']):05d}" / "samples"
        input_image_path = None
        if image_root is not None and image_key is not None:
            meta = rec.get("meta", {})
            if isinstance(meta, dict) and "meta" in meta:
                # Unified nested meta: {"meta": {<upstream row>}}
                meta = meta.get("meta", meta)
            rel = meta.get(image_key) if isinstance(meta, dict) else None
            if not rel:
                # Fall back to top-level record lookup.
                rel = rec.get(image_key)
            if not rel:
                raise KeyError(
                    f"record idx={rec.get('idx')} has no '{image_key}' field "
                    f"(looked in meta and record top-level)"
                )
            input_image_path = str(Path(image_root) / rel)
        for s in range(samples_per_prompt):
            png_path = sample_dir / f"{s:04d}.png"
            if resume and png_path.exists():
                continue
            job = {
                "idx": int(rec["idx"]),
                "sample_idx": s,
                "prompt": rec["prompt"],
                "meta": rec,
                "png_path": str(png_path),
            }
            if input_image_path is not None:
                job["input_image_path"] = input_image_path
            jobs.append(job)
    return jobs


def _write_metadata(run_dir: Path, prompts: List[dict]) -> None:
    """Write (or refresh) metadata.json for every prompt dir."""
    for rec in prompts:
        idx_dir = run_dir / f"{int(rec['idx']):05d}"
        idx_dir.mkdir(parents=True, exist_ok=True)
        with open(idx_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)


def _chunked(seq: Iterable, n: int):
    buf = []
    for item in seq:
        buf.append(item)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def _load_models(args, device):
    print(f"[rank] loading ar_model: {args.model_path}")
    ar_model = UniARForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn,
    ).to(device)
    ar_model.eval()

    ar_processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")
    visual_decoder = UniARVisualDecoder.from_pretrained(
        args.model_path,
        device=device,
    )

    return ar_model, ar_processor, visual_decoder


def main():
    parser = argparse.ArgumentParser(description="UniAR batched visual generation launcher")
    parser.add_argument("--model_path", required=True,
                        help="Local or HF path of packaged UniAR checkpoint")
    parser.add_argument("--data_path", required=True, help="jsonl with {idx, prompt, meta}")
    parser.add_argument("--output_path", default="eval/runs",
                        help="Parent dir; actual outputs land under <output_path>/<run_name>/")
    parser.add_argument("--run_name", required=True, help="Subdirectory name under output_path")
    parser.add_argument("--ar_height", type=int, default=512, help="AR visual-token rollout height")
    parser.add_argument("--ar_width", type=int, default=512, help="AR visual-token rollout width")
    parser.add_argument("--samples_per_prompt", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Rollout batch size per rank")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--decoder_num_inference_steps", type=int, default=28)
    parser.add_argument("--decoder_cfg_scale", type=float, default=1.5)
    parser.add_argument("--upsampling_ratio", type=float, default=2.0,
                        help="SD3 decoder upsampling ratio from AR resolution to output resolution.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--image_root", default=None,
                        help="If set together with --image_key, activates edit mode: "
                             "each prompt is conditioned on <image_root>/<meta[image_key]>.")
    parser.add_argument("--image_key", default=None,
                        help="Field in each record's `meta` (or top-level) holding the "
                             "input-image relative path. Requires --image_root.")
    args = parser.parse_args()

    if (args.image_root is None) != (args.image_key is None):
        parser.error("--image_root and --image_key must be set together (edit mode).")
    edit_mode = args.image_root is not None

    accelerator = Accelerator()
    device = accelerator.device

    ar_height = args.ar_height
    ar_width = args.ar_width
    upsampling_ratio = args.upsampling_ratio

    run_dir = Path(args.output_path) / args.run_name
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)

    prompts = _load_prompts(args.data_path)
    if accelerator.is_main_process:
        _write_metadata(run_dir, prompts)
        print(f"loaded {len(prompts)} prompts; samples_per_prompt={args.samples_per_prompt}")

    jobs = _build_jobs(
        prompts, args.samples_per_prompt, run_dir, resume=True,
        image_root=args.image_root, image_key=args.image_key,
    )
    if accelerator.is_main_process:
        total = len(prompts) * args.samples_per_prompt
        print(f"pending jobs: {len(jobs)} / {total}")
    if len(jobs) == 0:
        if accelerator.is_main_process:
            print("nothing to do; exiting.")
        return

    ar_model, ar_processor, visual_decoder = _load_models(args, device)

    accelerator.wait_for_everyone()

    # Shard jobs across ranks: rank r takes jobs[r::world_size]. We deliberately
    # avoid accelerator.prepare(dataloader) to keep the mapping from job index to
    # output path fully explicit (no padded/duplicated tail batches).
    my_jobs = jobs[accelerator.process_index :: accelerator.num_processes]
    if accelerator.is_main_process:
        print(f"rank 0 job count: {len(my_jobs)}; world_size={accelerator.num_processes}")

    # For reproducibility of SD3 denoising (AR multinomial is not seeded per-sample).
    base_seed = args.seed + accelerator.process_index * 1_000_003

    progress = tqdm(
        total=len(my_jobs),
        disable=not accelerator.is_main_process,
        desc="generating",
    )

    for batch in _chunked(my_jobs, args.batch_size):
        prompts_batch = [j["prompt"] for j in batch]
        input_images_batch = None
        if edit_mode:
            from PIL import Image
            # Resize every input image to the AR rollout resolution
            # ``(ar_width, ar_height)``. The model was trained on fixed-size
            # edit inputs, and Qwen2VLImageProcessor's smart_resize defaults
            # (min_pixels=4096, max_pixels=16M) let arbitrary-size images pass
            # through — so without this the vision encoder sees OOD grid_thw
            # (e.g. 64×48 instead of the trained 32×32), emitting visual tokens
            # outside the distribution the AR head was trained on.
            input_images_batch = [
                Image.open(j["input_image_path"]).convert("RGB").resize((ar_width, ar_height))
                for j in batch
            ]

        visual_inputs = prepare_visual_inputs(
            prompts_batch, ar_model, ar_processor, ar_height, ar_width,
            input_images=input_images_batch,
        )
        generated_visual_ids = ar_model.generate_visual(
            prefix_input_ids=visual_inputs["prefix_input_ids"],
            attention_mask=visual_inputs["attention_mask"],
            pos_ids_image=visual_inputs["pos_ids_image"],
            pos_ids_all=visual_inputs["pos_ids_all"],
            image_token_num=visual_inputs["image_token_num"],
            temperature=args.temperature,
            cfg=args.cfg,
            pixel_values=visual_inputs.get("pixel_values"),
            input_image_grid_thw=visual_inputs.get("input_image_grid_thw"),
            show_progress=accelerator.is_main_process,
        )

        generator = torch.Generator(device=device).manual_seed(base_seed)
        images = visual_decoder.decode(
            generated_visual_ids,
            ar_height=ar_height,
            ar_width=ar_width,
            upsampling_ratio=upsampling_ratio,
            num_inference_steps=args.decoder_num_inference_steps,
            guidance_scale=args.decoder_cfg_scale,
            generator=generator,
        )

        for job, img in zip(batch, images):
            png_path = Path(job["png_path"])
            png_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(png_path)
        progress.update(len(batch))

    progress.close()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Best-effort completeness check.
        missing = []
        for rec in prompts:
            sample_dir = run_dir / f"{int(rec['idx']):05d}" / "samples"
            for s in range(args.samples_per_prompt):
                if not (sample_dir / f"{s:04d}.png").exists():
                    missing.append((rec["idx"], s))
        if missing:
            print(f"WARNING: {len(missing)} samples missing; first few: {missing[:5]}")
        else:
            print(f"done. all samples rendered under {run_dir}")


if __name__ == "__main__":
    main()
