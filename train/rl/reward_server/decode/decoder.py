"""BSQ codes decoder using SD3 pipeline.

Converts BSQ visual token indices back to pixel images via the UniAR vision
encoder (for feature reconstruction) and a Stable Diffusion 3 pipeline.
"""

import os
import torch

from uniar.modeling_vision_decoder import (
    SD3Transformer2DModelWithSigLIP,
    StableDiffusion3PipelineWithSigLIP,
)
from uniar.vision_encoder import load_bsq_image_tokenizer_and_transform


class BSQDecoder:
    def __init__(
        self,
        image_tokenizer_path: str,
        sd3_transformer_path: str,
        sd3_path: str,
        image_width: int = 512,
        image_height: int = 512,
        num_inference_steps: int = 28,
        cfg_scale: float = 1.5,
        is_gt: bool = False,
        super_resolution: bool = False,
        upscale_factor: int = 1,
        inference_skip_final_layernorm: bool = False,
        device: torch.device = None,
    ):
        self.image_tokenizer = load_bsq_image_tokenizer_and_transform(
            image_tokenizer_path, resolution=None, feature_level=None, no_merger=False,
        ).to(device)

        transformer = SD3Transformer2DModelWithSigLIP.from_pretrained(
            sd3_transformer_path, torch_dtype=torch.bfloat16,
        )
        transformer.eval()

        self.pipeline = StableDiffusion3PipelineWithSigLIP.from_pretrained(
            sd3_path, transformer=transformer, torch_dtype=torch.bfloat16,
        ).to(device)

        self.super_resolution = super_resolution
        self.upscale_factor = upscale_factor
        self.inference_skip_final_layernorm = inference_skip_final_layernorm
        self.num_inference_steps = num_inference_steps
        self.cfg_scale = cfg_scale
        self.image_width = image_width
        self.image_height = image_height
        self.is_gt = is_gt
        self.device = device

    def decode_bsq_codes(
        self,
        bsq_codes,
        num_inference_steps: int = None,
        image_width: int = None,
        image_height: int = None,
        cfg_scale: float = None,
    ):
        bsq_codes = bsq_codes.to(self.device)
        bsq_features = self.image_tokenizer.bsq_indices_to_features(
            bsq_codes,
            spatial_merge_unit=4,
            inference_skip_final_layernorm=self.inference_skip_final_layernorm,
        )
        bsq_features = (
            torch.stack([bsq_features[0], *bsq_features[1]], dim=-1)
            .flatten(-2)
            .flatten(0, 1)
        )

        image_width = image_width or self.image_width
        image_height = image_height or self.image_height
        cfg_scale = cfg_scale or self.cfg_scale
        num_inference_steps = num_inference_steps or self.num_inference_steps

        merge_size = 1 if self.is_gt else 2
        grid_h, grid_w = image_height // 16, image_width // 16

        bsq_features = (
            bsq_features.view(
                grid_h // merge_size, grid_w // merge_size,
                merge_size, merge_size, -1,
            )
            .permute(0, 2, 1, 3, 4)
            .flatten(0, 3)
        )

        prompt = "Please reconstruct this image and restore all the details."
        if self.super_resolution:
            out_h = int(image_height * self.upscale_factor // 16) * 16
            out_w = int(image_width * self.upscale_factor // 16) * 16
        else:
            out_h = image_height
            out_w = image_width

        out_img = self.pipeline(
            siglip_tensor=bsq_features.unsqueeze(0),
            prompt=[prompt],
            negative_prompt=[prompt],
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_scale,
            generator=None,
            height=out_h,
            width=out_w,
        ).images[0]

        return out_img