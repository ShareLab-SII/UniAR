"""
Input preparation helpers for UniAR visual generation.
"""

from typing import List, Optional

import torch
from PIL import Image

from uniar import CHAT_TEMPLATE

try:
    from qwen_vl_utils import process_vision_info as _qwen_process_vision_info
except ImportError:  # pragma: no cover -- only needed for edit mode
    _qwen_process_vision_info = None


def _build_visual_messages(
    prompt: str,
    ar_height: int,
    ar_width: int,
    downsample_factor: int,
    input_image: Optional[Image.Image] = None,
):
    content = [{"type": "text", "text": prompt}]
    if input_image is not None:
        # Edit mode: input image goes between the instruction and the
        # "<image_gen> ..." marker. Both cond and uncond branches carry the
        # image — CFG is over text only, not over image conditioning.
        content.append({"type": "image", "image": input_image})
    content.append(
        {
            "type": "text",
            "text": (
                f"<image_gen> generate image "
                f"{ar_height // downsample_factor} {ar_width // downsample_factor}"
            ),
        }
    )
    return [{"role": "user", "content": content}]


def prepare_visual_inputs(
    prompts: List[str],
    ar_model,
    ar_processor,
    ar_height: int,
    ar_width: int,
    input_images: Optional[List[Image.Image]] = None,
):
    """Build a 2B left-padded prefix and mRoPE ids for visual generation.

    Returns a dict with tensor names expected by
    ``UniARForConditionalGeneration.generate_visual``:

    - ``prefix_input_ids``: conditional + unconditional prefix ids, shape ``(2B, L)``
    - ``attention_mask``: prefix attention mask, shape ``(2B, L)``
    - ``pos_ids_all``: mRoPE ids for prefix + generated visual tokens
    - ``pos_ids_image``: mRoPE ids for generated visual tokens only
    - ``image_token_num``: number of visual tokens to generate

    When ``input_images`` is provided, edit mode is enabled. Each prompt must
    have one PIL image. The image is carried by both the conditional and
    unconditional branches, so CFG only changes the text condition.
    """
    v_cfg = ar_model.config.vision_config
    patch = v_cfg.patch_size
    merge = v_cfg.spatial_merge_size
    downsample_factor = patch * merge

    is_edit = input_images is not None
    if is_edit:
        assert len(input_images) == len(prompts), (
            f"len(input_images) ({len(input_images)}) must match len(prompts) ({len(prompts)})"
        )
        if _qwen_process_vision_info is None:
            raise ImportError(
                "qwen_vl_utils is required for edit mode. "
                "Install via `pip install qwen-vl-utils`."
            )

    cond_messages = [
        _build_visual_messages(
            prompt,
            ar_height,
            ar_width,
            downsample_factor,
            input_image=(input_images[i] if is_edit else None),
        )
        for i, prompt in enumerate(prompts)
    ]
    uncond_messages = [
        _build_visual_messages(
            "",
            ar_height,
            ar_width,
            downsample_factor,
            input_image=(input_images[i] if is_edit else None),
        )
        for i in range(len(prompts))
    ]
    all_messages = cond_messages + uncond_messages

    ar_processor.chat_template = CHAT_TEMPLATE
    texts = ar_processor.apply_chat_template(
        all_messages, tokenize=False, add_generation_prompt=True
    )
    texts = [text + "<|vision_start|>" for text in texts]

    if is_edit:
        image_inputs, video_inputs = _qwen_process_vision_info(all_messages)
        inputs = ar_processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(ar_model.device)
    else:
        inputs = ar_processor(
            text=texts,
            padding=True,
            return_tensors="pt",
        ).to(ar_model.device)

    prefix_input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    batch_size_2 = prefix_input_ids.shape[0]
    image_token_num = (ar_height // downsample_factor) * (ar_width // downsample_factor)
    image_input_ids = torch.full(
        (batch_size_2, image_token_num),
        ar_model.config.image_token_id,
        device=ar_model.device,
        dtype=torch.long,
    )
    output_image_grid_thw = torch.tensor(
        [[1, ar_height // patch, ar_width // patch]] * batch_size_2,
        device=ar_model.device,
        dtype=torch.long,
    )
    attention_mask_all = torch.cat(
        [attention_mask, torch.ones_like(image_input_ids)], dim=1
    )

    if is_edit:
        # get_rope_index needs per-image grid_thw for BOTH the input image and
        # the output image. Interleave per sample:
        # [input_grid_b0, output_grid_b0, input_grid_b1, output_grid_b1, ...].
        input_image_grid_thw = inputs.image_grid_thw
        combined_rows = []
        for i in range(batch_size_2):
            combined_rows.append(input_image_grid_thw[i: i + 1])
            combined_rows.append(output_image_grid_thw[i: i + 1])
        combined_grid_thw = torch.cat(combined_rows, dim=0)
        pos_ids_all, _ = ar_model.model.get_rope_index(
            torch.cat([prefix_input_ids, image_input_ids], dim=1),
            combined_grid_thw,
            attention_mask=attention_mask_all,
        )
    else:
        pos_ids_all, _ = ar_model.model.get_rope_index(
            torch.cat([prefix_input_ids, image_input_ids], dim=1),
            output_image_grid_thw,
            attention_mask=attention_mask_all,
        )

    pos_ids_image = pos_ids_all[:, :, prefix_input_ids.shape[1]:]

    visual_inputs = {
        "prefix_input_ids": prefix_input_ids,
        "attention_mask": attention_mask,
        "pos_ids_all": pos_ids_all,
        "pos_ids_image": pos_ids_image,
        "image_token_num": image_token_num,
    }
    if is_edit:
        visual_inputs["pixel_values"] = inputs.pixel_values
        visual_inputs["input_image_grid_thw"] = inputs.image_grid_thw
    return visual_inputs