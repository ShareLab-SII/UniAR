"""
Simple UniAR image-understanding chat example.

This script loads a UniAR checkpoint, answers one question about one image, and
prints the generated text response.

Example:
    python inference/chat.py \
        --model_path /path/to/UniAR \
        --image https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
        --prompt "Describe this image in detail."
"""

import argparse
import os
import sys

import torch
from transformers import AutoProcessor

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from uniar import UniARForConditionalGeneration


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"loading model: {args.model_path}")
    attn_implementation = args.attn
    if device.type != "cuda" and attn_implementation == "flash_attention_2":
        attn_implementation = "eager"
        print("flash_attention_2 requires CUDA; using eager attention instead")

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = UniARForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        attn_implementation=attn_implementation,
    ).to(torch.bfloat16).to(device)
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("mm_token_type_ids", None)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print("output:", output_text[0] if output_text else "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple UniAR image-understanding chat example")
    parser.add_argument("--model_path", required=True, help="Path or HF id of a UniAR checkpoint")
    parser.add_argument("--image", required=True, help="Local path or URL of the input image")
    parser.add_argument("--prompt", required=True, help="Text prompt to ask about the image")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--attn", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    args = parser.parse_args()

    main(args)
