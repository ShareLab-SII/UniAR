"""
UniAR vision encoder: Qwen3-VL vision backbone + Binary Spherical Quantization.
"""

from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from .bsq import BinarySphericalQuantizer
from .configuration_vision_encoder import UniARVisionConfig


class UniARPatchMerger(nn.Module):
    """
    A Qwen3-VL-compatible patch merger with configurable middle dim.

    Parameter layout matches Qwen3VLVisionPatchMerger (norm, linear_fc1, linear_fc2),
    but supports an independent ``mid_size`` for the MLP bottleneck so the same
    module can act as the main merger or as a BSQ input/output projection.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        mid_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        if mid_size is None:
            mid_size = self.hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else context_dim
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, mid_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(mid_size, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape(-1, self.hidden_size))
        else:
            x = self.norm(x).reshape(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class UniARVisionModel(Qwen3VLVisionModel):
    """
    UniAR vision encoder.

    Extends ``Qwen3VLVisionModel`` with Binary Spherical Quantization (BSQ) between
    the transformer blocks and the patch merger, producing discrete visual tokens
    used by the UniAR autoregressive head.

    When ``config.use_bsq`` is ``False`` this behaves as a drop-in Qwen3-VL vision
    encoder.
    """

    config_class = UniARVisionConfig

    def __init__(self, config: UniARVisionConfig, *inputs, no_merger: bool = False, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)

        if no_merger:
            del self.merger

        if config.use_bsq:
            use_postshuffle_norm = not config.bsq_skip_final_layernorm
            if not no_merger:
                del self.merger
                self.merger = UniARPatchMerger(
                    dim=config.out_hidden_size,
                    context_dim=config.hidden_size,
                    spatial_merge_size=config.spatial_merge_size,
                    use_postshuffle_norm=use_postshuffle_norm,
                )
            self._init_bsq_params()
            for idx in range(len(config.deepstack_visual_indexes)):
                self._init_bsq_params(multiscale_idx=idx)

    def _init_bsq_params(self, multiscale_idx: Optional[int] = None) -> None:
        config = self.config
        bsq_dim = config.bsq_dim
        if multiscale_idx is None:
            self.bsq = BinarySphericalQuantizer(bsq_dim, input_format='c_last', gamma=0.5)

        bsq_input_proj = UniARPatchMerger(
            dim=bsq_dim,
            context_dim=config.hidden_size,
            spatial_merge_size=1,
            use_postshuffle_norm=True,
            mid_size=config.bsq_hidden_dim,
        )
        bsq_out_proj = UniARPatchMerger(
            dim=config.hidden_size,
            context_dim=bsq_dim,
            spatial_merge_size=1,
            use_postshuffle_norm=True,
            mid_size=config.bsq_hidden_dim,
        )

        input_name = 'bsq_input_proj' if multiscale_idx is None else f'bsq_input_proj_list_{multiscale_idx}'
        output_name = 'bsq_output_proj' if multiscale_idx is None else f'bsq_output_proj_list_{multiscale_idx}'
        setattr(self, input_name, bsq_input_proj)
        setattr(self, output_name, bsq_out_proj)

    def bsq_quantize(
        self,
        hidden_states: torch.Tensor,
        multiscale_idx: Optional[int] = None,
        direct_indices: bool = False,
        bsq_flip_prob: float = 0.0,
        bsq_flip_level: str = "per_batch",
    ) -> torch.Tensor:
        input_name = 'bsq_input_proj' if multiscale_idx is None else f'bsq_input_proj_list_{multiscale_idx}'
        output_name = 'bsq_output_proj' if multiscale_idx is None else f'bsq_output_proj_list_{multiscale_idx}'

        bsq_input_layer = getattr(self, input_name)
        bsq_output_layer = getattr(self, output_name)

        zq = bsq_input_layer(hidden_states)
        if isinstance(zq, tuple):
            zq = zq[0] + zq[1] if zq[1] is not None else zq[0]

        if bsq_flip_level == "per_bit" and bsq_flip_prob > 0.0:
            noise_apply_strength = np.random.randint(0, int(100 * bsq_flip_prob + 1)) * 0.01
            mask = (torch.rand(*zq.shape) < noise_apply_strength).to(zq)
            zq = zq * (1 - mask) + -zq * mask

        zq = F.normalize(zq, dim=-1)
        zq, _bsq_loss, meta = self.bsq(zq)

        bsq_indices = meta.get('indices', None)
        if bsq_indices is not None and not self.training:
            zq = self.bsq.get_codebook_entry(meta['indices']).bfloat16()
            if direct_indices:
                bsq_indices_gt = (self.bsq.indexes_to_codes(bsq_indices) + 1) // 2
                bsq_indices_gt = bsq_indices_gt.reshape(-1, bsq_indices_gt.shape[-1] * self.spatial_merge_unit)
                return bsq_indices_gt

        hidden_states = bsq_output_layer(zq)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0] + hidden_states[1] if hidden_states[1] is not None else hidden_states[0]
        return hidden_states

    def bsq_indices_to_features(
        self,
        vistok_pred: torch.Tensor,
        multiscale_idx='all',
        spatial_merge_unit: Optional[int] = None,
        with_merger: bool = False,
        inference_skip_final_layernorm: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if multiscale_idx == 'all':
            idx_list = [None] + list(range(len(self.config.deepstack_visual_indexes)))
        else:
            idx_list = [multiscale_idx]

        if spatial_merge_unit is None:
            spatial_merge_unit = self.spatial_merge_unit

        vistok_shape = (-1, len(idx_list), spatial_merge_unit, self.config.bsq_dim)
        vistok_pred = vistok_pred.view(*vistok_shape).bfloat16()
        zq = self.bsq.get_group_codebook_entry(vistok_pred).bfloat16()

        zq_list = [zq[:, i] for i in range(len(idx_list))] if multiscale_idx == 'all' else [zq]

        hidden_states: Optional[torch.Tensor] = None
        processed_deepstack_list: List[torch.Tensor] = []

        for i, ms_idx in enumerate(idx_list):
            zq_i = zq_list[i]
            output_name = 'bsq_output_proj' if ms_idx is None else f'bsq_output_proj_list_{ms_idx}'
            bsq_output_layer = getattr(self, output_name)
            hs = bsq_output_layer(zq_i)

            if (
                self.config.bsq_skip_final_layernorm
                and not inference_skip_final_layernorm
                and ms_idx is not None
            ):
                hs = self.merger.norm(hs)

            if ms_idx is None:
                if with_merger:
                    hs = self.merger(hs)
                hidden_states = hs
            else:
                if with_merger:
                    hs = self.deepstack_merger_list[ms_idx](hs)
                processed_deepstack_list.append(hs)

        return hidden_states, processed_deepstack_list

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        bsq_only: bool = False,
        bsq_feature_level: Optional[int] = None,
        direct_indices: bool = False,
        bsq_flip_prob: float = 0.0,
        bsq_flip_level: str = "per_batch",
        **kwargs,
    ):
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0],
        ).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        raw_deepstack_states: List[torch.Tensor] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                raw_deepstack_states.append(hidden_states)

        if self.config.use_bsq:
            hidden_states = self.bsq_quantize(
                hidden_states,
                multiscale_idx=None,
                direct_indices=direct_indices,
                bsq_flip_prob=bsq_flip_prob,
                bsq_flip_level=bsq_flip_level,
            )
            processed_deepstack: List[torch.Tensor] = []
            for idx, x in enumerate(raw_deepstack_states):
                x = self.bsq_quantize(
                    x,
                    multiscale_idx=idx,
                    direct_indices=direct_indices,
                    bsq_flip_prob=bsq_flip_prob,
                    bsq_flip_level=bsq_flip_level,
                )
                if not bsq_only and self.config.bsq_skip_final_layernorm:
                    x = self.merger.norm(x)
                processed_deepstack.append(x)
            raw_deepstack_states = processed_deepstack

            if bsq_only:
                if bsq_feature_level is None:
                    return hidden_states, processed_deepstack
                combined = [hidden_states] + processed_deepstack
                return combined[bsq_feature_level], []

        if bsq_only:
            return hidden_states, []

        hidden_states = self.merger(hidden_states)
        deepstack_feature_lists: List[torch.Tensor] = []
        for idx, x in enumerate(raw_deepstack_states):
            deepstack_feature_lists.append(self.deepstack_merger_list[idx](x))

        return hidden_states, deepstack_feature_lists


# ---------------------------------------------------------------------------
# BSQ image tokenizer loader
# ---------------------------------------------------------------------------


def _convert_img_to_patch(img: torch.Tensor):
    """Convert a BCHW image batch to the flattened patch layout UniAR expects."""
    spatial_patch_size = 16
    temporal_patch_size = 2
    spatial_merge_size = 1

    bs, c, h, w = img.shape
    grid_t = 1
    grid_h = h // spatial_merge_size // spatial_patch_size
    grid_w = w // spatial_merge_size // spatial_patch_size

    img = img.unsqueeze(2).expand(-1, -1, temporal_patch_size, -1, -1)
    flatten_image = img.reshape(
        bs,
        c,
        grid_t,
        temporal_patch_size,
        grid_h // spatial_merge_size,
        spatial_merge_size,
        spatial_patch_size,
        grid_w // spatial_merge_size,
        spatial_merge_size,
        spatial_patch_size,
    )
    flatten_image = flatten_image.permute(0, 2, 4, 7, 5, 8, 1, 3, 6, 9).reshape(
        bs * grid_t * grid_h * grid_w, -1
    )

    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]] * bs).to(flatten_image.device)
    return flatten_image, grid_thw


def load_bsq_image_tokenizer_and_transform(
    model_path: str,
    resolution: int | None = None,
    feature_level=None,
    no_merger: bool = False,
    subfolder: str | None = None,
):
    """
    Load a UniAR vision encoder as a BSQ image tokenizer.

    Args:
        model_path: Path or HF id of a saved UniARVisionModel checkpoint.
        resolution: Deprecated compatibility argument; loading is resolution-independent.
        feature_level: Which BSQ feature level to use (``None`` = all concatenated).
        no_merger: Skip the final merger (useful when only BSQ indices are needed).
        subfolder: Optional checkpoint subfolder, useful for packaged HF repos.

    Returns:
        The loaded ``UniARVisionModel`` with ``embed_dim`` and
        ``convert_img_to_patch`` attached for downstream consumers.
    """
    config_kwargs = {}
    if subfolder is not None:
        config_kwargs["subfolder"] = subfolder

    config = UniARVisionConfig.from_pretrained(model_path, **config_kwargs)

    model_kwargs = {
        "config": config,
        "torch_dtype": torch.bfloat16,
        "no_merger": no_merger,
        "ignore_mismatched_sizes": True,
    }
    if subfolder is not None:
        model_kwargs["subfolder"] = subfolder

    tokenizer = UniARVisionModel.from_pretrained(
        model_path,
        **model_kwargs,
    )
    tokenizer.is_moe = False
    tokenizer.convert_img_to_patch = _convert_img_to_patch

    if not tokenizer.config.deepstack_visual_indexes:
        tokenizer.embed_dim = tokenizer.config.hidden_size
    elif feature_level is None:
        tokenizer.embed_dim = tokenizer.config.hidden_size * (1 + len(tokenizer.config.deepstack_visual_indexes))
    elif feature_level in (0, 1, 2, 3):
        tokenizer.embed_dim = tokenizer.config.hidden_size
    else:
        raise NotImplementedError(f"Unsupported feature_level {feature_level}")

    return tokenizer


__all__ = [
    "UniARPatchMerger",
    "UniARVisionModel",
    "load_bsq_image_tokenizer_and_transform",
]
