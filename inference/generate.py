"""
Simple UniAR text-to-image generation example.

This script loads a packaged UniAR checkpoint, generates one image from one
prompt, and saves it as a PNG.

Example:
    python inference/generate.py \
        --model_path /path/to/UniAR \
        --prompt "A cute anime girl." \
        --output_path inference/generated.png
"""

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import AutoProcessor

from uniar import UniARForConditionalGeneration, UniARVisualDecoder

from inference.visual_inputs import prepare_visual_inputs

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"loading model: {args.model_path}")
    attn_implementation = args.attn

    ar_model = UniARForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    ).to(device)
    ar_model.eval()

    ar_processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")
    visual_decoder = UniARVisualDecoder.from_pretrained(args.model_path, device=device)

    ar_height = args.ar_height
    ar_width = args.ar_width
    upsampling_ratio = args.upsampling_ratio

    visual_inputs = prepare_visual_inputs(
        [args.prompt], ar_model, ar_processor,
        ar_height, ar_width,
    )
    generated_visual_ids = ar_model.generate_visual(
        prefix_input_ids=visual_inputs["prefix_input_ids"],
        attention_mask=visual_inputs["attention_mask"],
        pos_ids_image=visual_inputs["pos_ids_image"],
        pos_ids_all=visual_inputs["pos_ids_all"],
        image_token_num=visual_inputs["image_token_num"],
        temperature=args.temperature,
        cfg=args.cfg,
        show_progress=True,
    )

    images = visual_decoder.decode(
        generated_visual_ids,
        ar_height=ar_height,
        ar_width=ar_width,
        upsampling_ratio=upsampling_ratio,
        num_inference_steps=args.decoder_num_inference_steps,
        guidance_scale=args.decoder_cfg_scale,
    )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    images[0].save(args.output_path)
    print(f"saved image: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple UniAR text-to-image generation example")
    parser.add_argument("--model_path", required=True, help="Local or HF path of UniAR checkpoint")
    parser.add_argument("--prompt", required=True, help="a cute anime girl.")
    parser.add_argument("--output_path", default="inference/generated.png")
    parser.add_argument("--ar_height", type=int, default=960, help="AR visual-token rollout height")
    parser.add_argument("--ar_width", type=int, default=960, help="AR visual-token rollout width")
    parser.add_argument("--image_height", type=int, default=None,
                        help="Deprecated alias of --ar_height")
    parser.add_argument("--image_width", type=int, default=None,
                        help="Deprecated alias of --ar_width")
    parser.add_argument("--upsampling_ratio", type=float, default=1.067,
                        help="SD3 decoder upsampling ratio from AR resolution to output resolution.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--decoder_num_inference_steps", type=int, default=28)
    parser.add_argument("--decoder_cfg_scale", type=float, default=1.5)
    parser.add_argument("--attn", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    args = parser.parse_args()

    main(args)
