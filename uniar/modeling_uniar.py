"""UniAR model: Qwen3-VL backbone + BSQ vision encoder + visual token prediction head."""

from typing import Optional
from contextlib import contextmanager

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRMSNorm,
)

from uniar.configuration_uniar import UniARConfig
from uniar.vision_encoder.modeling_vision_encoder import UniARVisionModel


@contextmanager
def _skip_final_norm(text_model: nn.Module):
    """Swap ``text_model.norm`` for an ``nn.Identity`` for the duration of the block.

    During image-generation rollout, the ``output_layer_vistok`` head already
    applies its own RMSNorm as its first layer; stacking the backbone's final
    RMSNorm on top would differ from the training-time path. We temporarily
    bypass the backbone norm so the head sees the exact distribution it was
    trained on.
    """
    original = text_model.norm
    text_model.norm = nn.Identity()
    try:
        yield
    finally:
        text_model.norm = original


class UniARModel(Qwen3VLModel):
    """
    UniAR base model.

    Mirrors ``Qwen3VLModel`` but substitutes the vision tower with
    ``UniARVisionModel`` so BSQ-quantized visual tokens flow into the LLM.
    """

    config_class = UniARConfig

    def __init__(self, config: UniARConfig):
        super().__init__(config)
        del self.visual
        self.visual = UniARVisionModel._from_config(config.vision_config)


def _build_vistok_head(config: UniARConfig) -> nn.Module:
    vision_config = config.vision_config
    pred_level = 1 + len(vision_config.deepstack_visual_indexes)
    vispred_output_dim = (
        vision_config.bsq_dim * 2 * vision_config.spatial_merge_size ** 2 * pred_level
    )
    hidden_size = config.text_config.hidden_size

    if config.vistok_pred_use_mlp:
        middle_dim = 16 * 1024
        return nn.Sequential(
            Qwen3VLTextRMSNorm(hidden_size, eps=config.text_config.rms_norm_eps),
            nn.Linear(hidden_size, middle_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(middle_dim, vispred_output_dim, bias=False),
        )
    if config.vistok_pred_layernorm:
        return nn.Sequential(
            Qwen3VLTextRMSNorm(hidden_size, eps=config.text_config.rms_norm_eps),
            nn.Linear(hidden_size, vispred_output_dim, bias=False),
        )
    return nn.Linear(hidden_size, vispred_output_dim, bias=False)


class UniARForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """
    UniAR conditional generation model.

    Extends ``Qwen3VLForConditionalGeneration`` with:

    - ``model.visual`` swapped for ``UniARVisionModel`` (BSQ).
    - Optional ``output_layer_vistok`` head for visual-token (BSQ bit) prediction.
    - Optional ``visual_decoder`` — a small stack of text decoder layers applied
      on top of the LLM hidden states before the visual-token head.
    - ``generation_mode`` selector (``'text'`` vs ``'image'``) used by the
      generation loop to route through either ``lm_head`` or
      ``output_layer_vistok``.
    """

    config_class = UniARConfig

    def __init__(self, config: UniARConfig):
        Qwen3VLPreTrainedModel.__init__(self, config)
        GenerationMixin.__init__(self)
        self.model = UniARModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.rope_deltas = None

        if config.vistok_pred:
            self.output_layer_vistok = _build_vistok_head(config)
            if config.visual_transformer_decoder:
                self.visual_decoder = nn.ModuleList(
                    [
                        Qwen3VLTextDecoderLayer(config.text_config, layer_idx)
                        for layer_idx in range(config.visual_transformer_decoder_depth)
                    ]
                )

        self.generation_mode = "text"
        self.post_init()

    def set_generation_mode(self, mode: str) -> None:
        if mode not in ("text", "image"):
            raise ValueError(f"Invalid generation mode: {mode}")
        self.generation_mode = mode

    def visual_transformer_pred(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ret_all: bool = False,
    ) -> torch.Tensor:
        """Apply the optional ``visual_decoder`` stack before the vistok head."""
        if not getattr(self, "visual_decoder", None):
            raise RuntimeError(
                "visual_decoder is not configured; set config.visual_transformer_decoder=True."
            )

        if hidden_states.dim() == 2:
            hidden_states = hidden_states[None]
        if attention_mask is None:
            attention_mask = torch.ones_like(hidden_states[..., 0]).long()

        language_model = self.model.language_model
        position_embeddings = language_model.rotary_emb(hidden_states, position_ids)

        for layer in self.visual_decoder:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        if ret_all:
            return self.output_layer_vistok(hidden_states)
        return self.output_layer_vistok(hidden_states[:, -1, :])

    @torch.no_grad()
    def image_gen_step(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: bool = True,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """
        Run a single image-generation forward step through the LLM backbone.

        Returns ``(hidden_states_prenorm, past_key_values)``:
          - ``hidden_states_prenorm``: last-decoder-layer output BEFORE the final
            RMSNorm — the tensor the vistok_pred head was trained to consume.
          - ``past_key_values``: updated KV cache.

        Two call patterns:
          - Prefix step: pass ``input_ids`` (text prompt); the text embedder runs.
          - Incremental step: pass ``inputs_embeds`` (pre-baked BSQ feature) plus
            ``deepstack_visual_embeds`` (list of per-deepstack-layer features);
            a visual_pos_masks of all-ones is attached automatically.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("image_gen_step requires either input_ids or inputs_embeds")

        visual_pos_masks = None
        if inputs_embeds is not None:
            if inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds[:, None, :]
            visual_pos_masks = inputs_embeds.new_ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]), dtype=torch.bool
            )

        text_model = self.model.language_model
        if inputs_embeds is None:
            inputs_embeds = text_model.embed_tokens(input_ids)

        with _skip_final_norm(text_model):
            outputs = text_model(
                inputs_embeds=inputs_embeds,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        return outputs.last_hidden_state, outputs.past_key_values

    @torch.no_grad()
    def image_gen_step_with_input_image(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: bool = True,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """
        Image-conditioned prefill step for the edit-mode rollout.

        Runs ``UniARModel`` end-to-end (vision encoder → image-feature splice
        into ``inputs_embeds`` → text backbone) with the backbone's final
        RMSNorm skipped — the same pre-norm exit ``image_gen_step`` uses — so
        the downstream ``output_layer_vistok`` head sees its trained input.

        Use this for the i==0 step in edit mode; from i>=1 onward, fall back to
        ``image_gen_step(inputs_embeds=..., deepstack_visual_embeds=...)``
        exactly as in T2I rollout.
        """
        text_model = self.model.language_model
        with _skip_final_norm(text_model):
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
            )
        return outputs.last_hidden_state, outputs.past_key_values

    @torch.no_grad()
    def generate_visual(
        self,
        prefix_input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pos_ids_image: torch.LongTensor,
        pos_ids_all: torch.LongTensor,
        image_token_num: int,
        temperature: float,
        cfg: float = 1.0,
        pixel_values: Optional[torch.Tensor] = None,
        input_image_grid_thw: Optional[torch.LongTensor] = None,
        show_progress: bool = False,
        return_raw_codes: bool = False,
    ) -> torch.Tensor:
        """Autoregressively generate BSQ visual-token indices.

        This is UniAR's canonical visual generation API, shared by inference
        and RL rollout.

        When ``cfg > 1.0``, the input batch is assumed to be CFG-expanded
        (first half conditional, second half unconditional). The returned
        tensor keeps only the conditional half.

        When ``cfg <= 1.0``, no CFG is applied; the full batch is used as-is.

        Returns integer indices by default; pass ``return_raw_codes=True``
        to get raw binary codes (0/1) instead (used by RL training).
        """
        if image_token_num <= 0:
            raise ValueError("image_token_num must be > 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if pixel_values is not None and input_image_grid_thw is None:
            raise ValueError(
                "input_image_grid_thw is required when pixel_values is provided"
            )
        if pixel_values is None and input_image_grid_thw is not None:
            raise ValueError(
                "pixel_values is required when input_image_grid_thw is provided"
            )

        if not getattr(self.config, "visual_transformer_decoder", False):
            raise NotImplementedError(
                "generate_visual currently requires visual_transformer_decoder=True"
            )

        use_cfg = cfg > 1.0
        vision_cfg = self.visual.config
        num_deepstack = len(vision_cfg.deepstack_visual_indexes)
        spatial_merge_unit = self.visual.spatial_merge_unit

        generated = []
        hidden_state_list = []
        past_key_values = None
        inputs_embeds = None
        deepstack_visual_embeds = None
        ori_hidden_state = None
        incremental_attention_mask = attention_mask

        iterator = range(image_token_num)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="AR rollout", leave=False)

        for i in iterator:
            if i > 0:
                ones = torch.ones(
                    (incremental_attention_mask.shape[0], 1),
                    device=incremental_attention_mask.device,
                    dtype=incremental_attention_mask.dtype,
                )
                incremental_attention_mask = torch.cat(
                    [incremental_attention_mask, ones], dim=1
                )

            if i == 0:
                pos_ids_prefix = pos_ids_all[:, :, : prefix_input_ids.shape[1]]
                if pixel_values is not None:
                    hidden_states, past_key_values = self.image_gen_step_with_input_image(
                        input_ids=prefix_input_ids,
                        pixel_values=pixel_values,
                        image_grid_thw=input_image_grid_thw,
                        attention_mask=incremental_attention_mask,
                        position_ids=pos_ids_prefix,
                        past_key_values=None,
                        use_cache=True,
                    )
                else:
                    hidden_states, past_key_values = self.image_gen_step(
                        input_ids=prefix_input_ids,
                        attention_mask=incremental_attention_mask,
                        position_ids=pos_ids_prefix,
                        past_key_values=None,
                        use_cache=True,
                    )
                ori_hidden_state = hidden_states
            else:
                hidden_states, past_key_values = self.image_gen_step(
                    inputs_embeds=inputs_embeds,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                    attention_mask=incremental_attention_mask,
                    position_ids=pos_ids_image[:, :, i - 1: i],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            hidden_state_list.append(hidden_states[:, -1, :].unsqueeze(1))

            generated_hidden_states = torch.cat(hidden_state_list, dim=1)
            combined_hidden_states = torch.cat([ori_hidden_state[:, :-1, :], generated_hidden_states], dim=1)
            pos_ids_refine = pos_ids_all[..., : combined_hidden_states.shape[1]]
            vistok_logits = self.visual_transformer_pred(
                combined_hidden_states,
                position_ids=pos_ids_refine,
                attention_mask=incremental_attention_mask,
            )

            if use_cfg:
                bs = vistok_logits.shape[0]
                cond_logits = vistok_logits[: bs // 2]
                uncond_logits = vistok_logits[bs // 2:]
                vistok_logits = uncond_logits + cfg * (cond_logits - uncond_logits)
                vistok_logits = torch.cat([vistok_logits, vistok_logits], dim=0)

            num_levels = 1 + num_deepstack
            vistok_shape = (-1, num_levels, spatial_merge_unit, vision_cfg.bsq_dim, 2)
            vistok_logits = vistok_logits.view(vistok_shape)
            vistok_logits = vistok_logits / temperature
            vistok_pred = torch.multinomial(
                vistok_logits.flatten(0, -2).softmax(-1), num_samples=1
            )
            vistok_pred = vistok_pred.view(vistok_logits.shape[:-1])

            (
                inputs_embeds_main,
                deepstack_visual_embeds,
            ) = self.visual.bsq_indices_to_features(
                vistok_pred.to(self.device),
                spatial_merge_unit=spatial_merge_unit,
                with_merger=True,
            )
            inputs_embeds = inputs_embeds_main
            generated.append(vistok_pred)

        vistok_all = torch.stack(generated).permute(1, 0, 2, 3, 4)

        if return_raw_codes:
            if use_cfg:
                bs = vistok_all.shape[0]
                return vistok_all[: bs // 2]
            return vistok_all

        vistok_all = vistok_all * 2 - 1  # {0,1} -> {-1,+1}
        if use_cfg:
            bs = vistok_all.shape[0]
            cond_codes = vistok_all[: bs // 2].to(self.device)
        else:
            cond_codes = vistok_all.to(self.device)
        return self.visual.bsq.codes_to_indexes(cond_codes)

    @torch.no_grad()
    def generate_rollout(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        generation_config=None,
        **kwargs,
    ) -> torch.Tensor:
        """RL rollout wrapper: prepare inputs and call ``generate_visual()``.

        Takes the same interface as HuggingFace ``generate()`` (input_ids +
        attention_mask + generation_config) so the RL trainer can call it with
        minimal changes.  Internally computes position IDs and image token
        count from ``self.image_height`` / ``self.image_width`` (set by the
        training script), then delegates to ``generate_visual(cfg=1.0,
        return_raw_codes=True)``.

        Returns raw BSQ codes with shape
        ``(B, image_token_num, num_levels, spatial_merge_unit, bsq_dim)``.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        v_cfg = self.config.vision_config
        patch = v_cfg.patch_size
        merge = v_cfg.spatial_merge_size
        downsample = patch * merge

        image_h = self.image_height
        image_w = self.image_width
        image_token_num = (image_h // downsample) * (image_w // downsample)

        image_grid_thw = torch.tensor(
            [1, image_h // patch, image_w // patch],
            device=device, dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)
        image_input_ids = torch.full(
            (batch_size, image_token_num), self.config.image_token_id,
            device=device, dtype=torch.long,
        )

        all_ids = torch.cat([input_ids, image_input_ids], dim=1)
        all_mask = torch.cat([
            attention_mask,
            torch.ones_like(image_input_ids),
        ], dim=1)
        pos_ids_all, _ = self.model.get_rope_index(all_ids, image_grid_thw, attention_mask=all_mask)
        pos_ids_image = pos_ids_all[:, :, input_ids.shape[1]:]

        temperature = 1.0
        if generation_config is not None:
            temperature = getattr(generation_config, "temperature", 1.0)

        return self.generate_visual(
            prefix_input_ids=input_ids,
            attention_mask=attention_mask,
            pos_ids_image=pos_ids_image,
            pos_ids_all=pos_ids_all,
            image_token_num=image_token_num,
            temperature=temperature,
            cfg=1.0,
            return_raw_codes=True,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        second_per_grid_ts=None,
        is_vistok_pred: bool = False,
        online_rollout_codes=None,
    ):
        """Forward pass with optional visual-token prediction for RL training.

        When ``is_vistok_pred=False`` (default), delegates to the standard
        Qwen3VL forward — text generation, understanding, etc. work as usual.

        When ``is_vistok_pred=True``, runs the RL loss-computation path:
        embeds ``online_rollout_codes`` at image-token positions, runs the LLM
        with final-norm skipped, and returns vistok logits via
        ``output_layer_vistok``.
        """
        if not is_vistok_pred:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                cache_position=cache_position,
                second_per_grid_ts=second_per_grid_ts,
            )

        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLCausalLMOutputWithPast,
        )

        inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds, image_embeds_multiscale = self.visual.bsq_indices_to_features(
            online_rollout_codes,
            spatial_merge_unit=self.visual.spatial_merge_unit,
            with_merger=True,
        )

        mask = input_ids == self.config.image_token_id
        mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, image_embeds)

        visual_pos_masks = mask
        deepstack_visual_embeds_multiscale = image_embeds_multiscale

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids, image_grid_thw, None, attention_mask,
        )

        text_model = self.model.language_model
        with _skip_final_norm(text_model):
            outputs = text_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                deepstack_visual_embeds=deepstack_visual_embeds_multiscale,
                visual_pos_masks=visual_pos_masks,
            )

        hidden_states = outputs.last_hidden_state

        if getattr(self.config, 'visual_transformer_decoder', False):
            vispred_logits = self.visual_transformer_pred(
                hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, ret_all=True,
            )
            image_mask = mask
            image_mask = torch.cat((image_mask[:, 1:], image_mask[:, :1]), dim=1)
            vispred_logits = vispred_logits[image_mask].reshape(
                input_ids.shape[0], -1, vispred_logits.shape[-1],
            )
        else:
            image_mask = mask
            image_mask = torch.cat((image_mask[:, 1:], image_mask[:, :1]), dim=1)
            vispred_hidden = hidden_states[image_mask].reshape(
                input_ids.shape[0], -1, hidden_states.shape[-1],
            )
            vispred_logits = self.output_layer_vistok(vispred_hidden)

        idx_list = [None] + list(range(len(self.visual.config.deepstack_visual_indexes)))
        vistok_shape = (
            input_ids.shape[0], -1, len(idx_list),
            self.visual.spatial_merge_unit, self.visual.config.bsq_dim, 2,
        )
        vispred_logits = vispred_logits.view(vistok_shape)

        return Qwen3VLCausalLMOutputWithPast(
            loss=None,
            logits=vispred_logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "UniARModel",
    "UniARForConditionalGeneration",
]
