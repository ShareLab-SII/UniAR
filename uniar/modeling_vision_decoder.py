"""
UniAR vision decoder: SD3 + SigLIP/BSQ conditioning, and high-level decode API.

This module provides:

- ``SD3Transformer2DModelWithSigLIP`` — ``diffusers.SD3Transformer2DModel``
  with an extra ``siglip_tensor`` conditioning path (additive on latent).
- ``StableDiffusion3PipelineWithSigLIP`` — SD3 pipeline whose ``__call__``
  threads the BSQ features through the transformer during denoising.
- ``UniARVisualDecoder`` — high-level wrapper that loads a packaged BSQ
  encoder + SD3 pipeline and decodes visual-token indices into PIL images.

Weights are stored as standard diffusers checkpoints (``config.json`` +
``diffusion_pytorch_model.safetensors``) — load with ``from_pretrained``.
"""

import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image

from diffusers import SD3Transformer2DModel
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import register_to_config
from diffusers.image_processor import PipelineImageInput
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_output import (
    StableDiffusion3PipelineOutput,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# SD3 Transformer with SigLIP/BSQ feature injection
# ---------------------------------------------------------------------------


class SD3Transformer2DModelWithSigLIP(SD3Transformer2DModel):
    """SD3 Transformer with additive BSQ/SigLIP feature conditioning.

    Extends the standard ``SD3Transformer2DModel`` with a learnable linear
    projection that maps BSQ encoder features to the DiT hidden dim and adds
    them to the patchified latent before transformer blocks.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: Tuple[int, ...] = (),
        qk_norm: Optional[str] = None,
        # BSQ/SigLIP conditioning
        siglip_channels: Optional[int] = None,
        drop_image_token_prob_for_cfg: float = 0.0,
        image_token_scale_factor: float = 1.0,
        add_siglip_tensor_to_latent: bool = True,
        # Super-resolution (interpolate upsampling)
        super_resolution: bool = False,
        upscale_factor: int = 1,
        # Random crop positional embedding (training only)
        train_use_random_crop_pos_embed: bool = False,
        random_crop_pos_embed_max_resolution: Optional[int] = None,
    ):
        super().__init__(
            sample_size=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            caption_projection_dim=caption_projection_dim,
            pooled_projection_dim=pooled_projection_dim,
            out_channels=out_channels,
            pos_embed_max_size=pos_embed_max_size,
            dual_attention_layers=dual_attention_layers,
            qk_norm=qk_norm,
        )

        self.siglip_channels = siglip_channels
        self.drop_image_token_prob_for_cfg = drop_image_token_prob_for_cfg
        self.image_token_scale_factor = image_token_scale_factor
        self.add_siglip_tensor_to_latent = add_siglip_tensor_to_latent

        if self.add_siglip_tensor_to_latent:
            self.siglip_embed_add_latent = nn.Linear(
                self.siglip_channels, self.inner_dim, bias=False
            )
            self.siglip_embed_add_latent.to_empty(device="cpu")
            nn.init.zeros_(self.siglip_embed_add_latent.weight)

        # Super-resolution via bicubic interpolation
        self.super_resolution = super_resolution
        self.upscale_factor = upscale_factor

        # Random crop positional embedding (training only)
        self.train_use_random_crop_pos_embed = train_use_random_crop_pos_embed
        if random_crop_pos_embed_max_resolution is not None:
            self.random_crop_pos_embed_max_grid = (
                random_crop_pos_embed_max_resolution // 8 // patch_size
            )
        else:
            self.random_crop_pos_embed_max_grid = None

    def prepare_siglip_tensor(
        self,
        siglip_tensor: Optional[torch.Tensor],
        target_spatial_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[torch.Tensor]:
        """Project and optionally upsample BSQ features for latent injection.

        Args:
            siglip_tensor: BSQ features, shape ``(B, seq_len, C)``.
            target_spatial_size: ``(target_h, target_w)`` after upsampling.

        Returns:
            Projected features ready to add to hidden states, or ``None``.
        """
        if siglip_tensor is None:
            return None

        # CFG dropout during training
        if self.drop_image_token_prob_for_cfg > 0 and self.training:
            keep_prob = 1 - self.drop_image_token_prob_for_cfg
            shape = (siglip_tensor.shape[0], 1, 1)
            random_tensor = siglip_tensor.new_empty(shape).bernoulli_(keep_prob)
            siglip_tensor = siglip_tensor * random_tensor

        if not self.add_siglip_tensor_to_latent:
            return None

        # Upsample if super-resolution is enabled
        if self.super_resolution:
            bs, seq_len, channel = siglip_tensor.shape

            if (
                target_spatial_size is not None
                and target_spatial_size[0] * target_spatial_size[1] == seq_len
            ):
                grid_h, grid_w = target_spatial_size
            else:
                grid_size = int(seq_len**0.5)
                grid_h, grid_w = grid_size, grid_size

            siglip_tensor = siglip_tensor.reshape(bs, grid_h, grid_w, channel)
            siglip_tensor = siglip_tensor.permute(0, 3, 1, 2)

            if target_spatial_size is not None:
                target_h, target_w = target_spatial_size
            else:
                target_h = grid_h * int(self.upscale_factor)
                target_w = grid_w * int(self.upscale_factor)

            siglip_tensor = torch.nn.functional.interpolate(
                siglip_tensor,
                size=(target_h, target_w),
                mode="bicubic",
                align_corners=False,
            )
            siglip_tensor = siglip_tensor.permute(0, 2, 3, 1)
            siglip_tensor = siglip_tensor.reshape(bs, -1, channel)

        return self.siglip_embed_add_latent(siglip_tensor)

    # ------------------------------------------------------------------
    # Random crop positional embedding helpers (training only)
    # ------------------------------------------------------------------

    def _sample_random_crop_offset(self, height: int, width: int) -> Tuple[int, int]:
        """Sample random crop offset for positional embeddings during training."""
        patch_size = self.pos_embed.patch_size
        pos_embed_max_size = self.pos_embed.pos_embed_max_size

        grid_h = height // patch_size
        grid_w = width // patch_size

        if self.random_crop_pos_embed_max_grid is not None:
            effective_max_size = min(
                self.random_crop_pos_embed_max_grid, pos_embed_max_size
            )
        else:
            effective_max_size = pos_embed_max_size

        center_offset_top = (pos_embed_max_size - effective_max_size) // 2
        center_offset_left = (pos_embed_max_size - effective_max_size) // 2

        max_top = max(0, effective_max_size - grid_h)
        max_left = max(0, effective_max_size - grid_w)

        top = center_offset_top + random.randint(0, max_top)
        left = center_offset_left + random.randint(0, max_left)

        return (top, left)

    def _get_cropped_pos_embed(
        self, height: int, width: int, crop_offset: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Get cropped positional embeddings (random or center crop)."""
        patch_size = self.pos_embed.patch_size
        pos_embed_max_size = self.pos_embed.pos_embed_max_size

        grid_h = height // patch_size
        grid_w = width // patch_size

        if crop_offset is None:
            top = (pos_embed_max_size - grid_h) // 2
            left = (pos_embed_max_size - grid_w) // 2
        else:
            top, left = crop_offset

        spatial_pos_embed = self.pos_embed.pos_embed.reshape(
            1, pos_embed_max_size, pos_embed_max_size, -1
        )
        spatial_pos_embed = spatial_pos_embed[
            :, top : top + grid_h, left : left + grid_w, :
        ]
        return spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])

    def _apply_pos_embed(
        self, latent: torch.Tensor, crop_offset: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Apply patch embedding + positional embedding with optional random crop."""
        height, width = latent.shape[-2:]

        latent = self.pos_embed.proj(latent)
        if self.pos_embed.flatten:
            latent = latent.flatten(2).transpose(1, 2)
        if self.pos_embed.layer_norm:
            latent = self.pos_embed.norm(latent)

        if self.pos_embed.pos_embed is None:
            return latent.to(latent.dtype)

        pos_embed = self._get_cropped_pos_embed(height, width, crop_offset)
        return (latent + pos_embed).to(latent.dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: Optional[List] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        siglip_tensor: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if (
                joint_attention_kwargs is not None
                and joint_attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using "
                    "the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]

        # Compute target spatial size for siglip upsampling
        patch_size = self.config.patch_size
        target_spatial_size = (height // patch_size, width // patch_size)

        # Process siglip tensor
        scaled_siglip = (
            siglip_tensor * self.image_token_scale_factor
            if siglip_tensor is not None
            else None
        )
        siglip_hidden_states = self.prepare_siglip_tensor(
            scaled_siglip, target_spatial_size=target_spatial_size
        )

        # Apply patch embedding + positional embedding
        if self.train_use_random_crop_pos_embed and self.training:
            crop_offset = self._sample_random_crop_offset(height, width)
            hidden_states = self._apply_pos_embed(hidden_states, crop_offset)
        else:
            hidden_states = self.pos_embed(hidden_states)

        # Text conditioning
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Inject BSQ features into latent
        if siglip_hidden_states is not None:
            hidden_states = hidden_states + siglip_hidden_states

        if (
            joint_attention_kwargs is not None
            and "ip_adapter_image_embeds" in joint_attention_kwargs
        ):
            ip_adapter_image_embeds = joint_attention_kwargs.pop(
                "ip_adapter_image_embeds"
            )
            ip_hidden_states, ip_temb = self.image_proj(
                ip_adapter_image_embeds, timestep
            )
            joint_attention_kwargs.update(
                ip_hidden_states=ip_hidden_states, temb=ip_temb
            )

        for index_block, block in enumerate(self.transformer_blocks):
            is_skip = (
                skip_layers is not None and index_block in skip_layers
            )

            if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:
                encoder_hidden_states, hidden_states = (
                    self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        joint_attention_kwargs,
                    )
                )
            elif not is_skip:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            if (
                block_controlnet_hidden_states is not None
                and block.context_pre_only is False
            ):
                interval_control = len(self.transformer_blocks) / len(
                    block_controlnet_hidden_states
                )
                hidden_states = hidden_states + block_controlnet_hidden_states[
                    int(index_block / interval_control)
                ]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            hidden_states.shape[0],
            height,
            width,
            patch_size,
            patch_size,
            self.out_channels,
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            hidden_states.shape[0],
            self.out_channels,
            height * patch_size,
            width * patch_size,
        )

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


# ---------------------------------------------------------------------------
# SD3 Pipeline with SigLIP/BSQ conditioning
# ---------------------------------------------------------------------------


class StableDiffusion3PipelineWithSigLIP(StableDiffusion3Pipeline):
    """SD3 pipeline that threads BSQ features through the transformer."""

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_guidance_layers: Optional[List[int]] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
        siglip_tensor: Optional[torch.Tensor] = None,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                original_prompt_embeds = prompt_embeds
                original_pooled_prompt_embeds = pooled_prompt_embeds
            prompt_embeds = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0
            )
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, h_lat, w_lat = latents.shape
            image_seq_len = (h_lat // self.transformer.config.patch_size) * (
                w_lat // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 6. Prepare image embeddings (IP-Adapter)
        if (
            ip_adapter_image is not None and self.is_ip_adapter_active
        ) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {
                    "ip_adapter_image_embeds": ip_adapter_image_embeds
                }
            else:
                self._joint_attention_kwargs.update(
                    ip_adapter_image_embeds=ip_adapter_image_embeds
                )

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                timestep = t.expand(latent_model_input.shape[0])

                # Prepare siglip_tensor for CFG
                if siglip_tensor is not None:
                    if self.do_classifier_free_guidance:
                        siglip_tensor_input = torch.cat(
                            [torch.zeros_like(siglip_tensor), siglip_tensor], dim=0
                        )
                    else:
                        siglip_tensor_input = siglip_tensor
                else:
                    siglip_tensor_input = None

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    siglip_tensor=siglip_tensor_input,
                )[0]

                # Perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    should_skip_layers = (
                        i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
                    )
                    if skip_guidance_layers is not None and should_skip_layers:
                        timestep = t.expand(latents.shape[0])
                        latent_model_input = latents
                        noise_pred_skip_layers = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=original_prompt_embeds,
                            pooled_projections=original_pooled_prompt_embeds,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                            skip_layers=skip_guidance_layers,
                        )[0]
                        noise_pred = noise_pred + (
                            noise_pred_text - noise_pred_skip_layers
                        ) * self._skip_layer_guidance_scale

                # Compute x_t -> x_{t-1}
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop(
                        "prompt_embeds", prompt_embeds
                    )
                    pooled_prompt_embeds = callback_outputs.pop(
                        "pooled_prompt_embeds", pooled_prompt_embeds
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents
        else:
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)


# ---------------------------------------------------------------------------
# High-level visual decoder (BSQ encoder + SD3 pipeline)
# ---------------------------------------------------------------------------


def _resolve_checkpoint_subfolder(model_path: str, subfolder: str):
    local_subfolder = os.path.join(model_path, subfolder)
    if os.path.isdir(local_subfolder):
        return local_subfolder, None
    if os.path.isdir(model_path):
        return model_path, None
    return model_path, subfolder


class UniARVisualDecoder:
    """Decode UniAR visual-token indices with a packaged BSQ encoder + SD3 decoder.

    Typical usage::

        decoder = UniARVisualDecoder.from_pretrained("path/to/UniAR", device="cuda")
        images = decoder.decode(visual_ids, ar_height=1024, ar_width=1024)
    """

    def __init__(self, bsq_encoder, pipeline):
        self.bsq_encoder = bsq_encoder
        self.pipeline = pipeline

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device=None,
        torch_dtype: torch.dtype = torch.bfloat16,
        bsq_encoder_path: Optional[str] = None,
        sd3_transformer_path: Optional[str] = None,
        sd3_pipeline_path: Optional[str] = None,
    ):
        """Load visual decoder components packaged under a UniAR checkpoint.

        Default layout::

            model_path/bsq_encoder
            model_path/sd3_transformer
            model_path/sd3_pipeline
        """
        from uniar.vision_encoder import load_bsq_image_tokenizer_and_transform

        bsq_encoder_path, bsq_encoder_subfolder = _resolve_checkpoint_subfolder(
            bsq_encoder_path or model_path, "bsq_encoder"
        )
        sd3_transformer_path, sd3_transformer_subfolder = (
            _resolve_checkpoint_subfolder(
                sd3_transformer_path or model_path, "sd3_transformer"
            )
        )
        sd3_pipeline_path, sd3_pipeline_subfolder = _resolve_checkpoint_subfolder(
            sd3_pipeline_path or model_path, "sd3_pipeline"
        )

        print(f"loading bsq_encoder: {bsq_encoder_path}")
        bsq_encoder = load_bsq_image_tokenizer_and_transform(
            bsq_encoder_path,
            resolution=None,
            feature_level=None,
            no_merger=False,
            subfolder=bsq_encoder_subfolder,
        )
        if device is not None:
            bsq_encoder = bsq_encoder.to(device)
        bsq_encoder.eval()

        print(f"loading SD3 transformer: {sd3_transformer_path}")
        transformer_kwargs = {"torch_dtype": torch_dtype}
        if sd3_transformer_subfolder is not None:
            transformer_kwargs["subfolder"] = sd3_transformer_subfolder
        transformer = SD3Transformer2DModelWithSigLIP.from_pretrained(
            sd3_transformer_path, **transformer_kwargs
        )
        transformer.eval()

        print(f"loading SD3 pipeline base: {sd3_pipeline_path}")
        pipeline_kwargs = {"transformer": transformer, "torch_dtype": torch_dtype}
        if sd3_pipeline_subfolder is not None:
            pipeline_kwargs["subfolder"] = sd3_pipeline_subfolder
        pipeline = StableDiffusion3PipelineWithSigLIP.from_pretrained(
            sd3_pipeline_path, **pipeline_kwargs
        )
        if device is not None:
            pipeline = pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)

        return cls(bsq_encoder=bsq_encoder, pipeline=pipeline)

    @torch.no_grad()
    def decode(
        self,
        visual_generated_ids: torch.Tensor,
        ar_height: int,
        ar_width: int,
        upsampling_ratio: float = 2.0,
        decoder_prompt: str = "Please reconstruct this image and restore all the details.",
        num_inference_steps: int = 28,
        guidance_scale: float = 1.5,
        generator=None,
    ) -> List[Image.Image]:
        """Decode BSQ visual-token indices into PIL images.

        Args:
            visual_generated_ids: BSQ indices from ``generate_visual()``.
            ar_height: AR rollout height.
            ar_width: AR rollout width.
            upsampling_ratio: SD3 decoder output size multiplier.
            decoder_prompt: Text prompt for the SD3 decoder.
            num_inference_steps: SD3 denoising steps.
            guidance_scale: SD3 CFG scale.
            generator: Torch generator for reproducibility.

        Returns:
            List of PIL images.
        """
        target_height = int(round(ar_height * upsampling_ratio))
        target_width = int(round(ar_width * upsampling_ratio))

        bsq_encoder = self.bsq_encoder
        pipeline = self.pipeline
        device = bsq_encoder.device
        visual_generated_ids = visual_generated_ids.to(device)
        batch_size = visual_generated_ids.shape[0]

        codes = bsq_encoder.bsq.indexes_to_codes(visual_generated_ids)
        codes = (codes + 1) // 2

        main_feat, deepstack_feats = bsq_encoder.bsq_indices_to_features(
            codes,
            spatial_merge_unit=bsq_encoder.spatial_merge_unit,
            inference_skip_final_layernorm=True,
        )
        features = torch.stack([main_feat, *deepstack_feats], dim=-1).flatten(-2)

        spatial_merge_unit = bsq_encoder.spatial_merge_unit
        channels = features.shape[-1]
        features = features.view(batch_size, -1, spatial_merge_unit, channels)

        merge = bsq_encoder.config.spatial_merge_size
        grid_h, grid_w = ar_height // 16, ar_width // 16
        features = features.view(
            batch_size, grid_h // merge, grid_w // merge, merge, merge, channels
        )
        features = features.permute(0, 1, 3, 2, 4, 5).flatten(1, 4)
        siglip_tensor = features.to(pipeline.transformer.dtype)

        return pipeline(
            siglip_tensor=siglip_tensor,
            prompt=[decoder_prompt] * batch_size,
            negative_prompt=[decoder_prompt] * batch_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=target_height,
            width=target_width,
        ).images


__all__ = [
    "SD3Transformer2DModelWithSigLIP",
    "StableDiffusion3PipelineWithSigLIP",
    "UniARVisualDecoder",
]
