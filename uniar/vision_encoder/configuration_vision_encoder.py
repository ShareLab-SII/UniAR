"""
UniAR vision encoder configuration (Qwen3-VL vision backbone + BSQ).
"""

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig


class UniARVisionConfig(Qwen3VLVisionConfig):
    """
    Configuration for UniAR vision encoder.

    Extends Qwen3VLVisionConfig with Binary Spherical Quantization (BSQ)
    fields used by UniAR to compress visual tokens into a discrete codebook.
    """

    model_type = "uniar_vision"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=4096,
        num_position_embeddings=2304,
        deepstack_visual_indexes=(8, 16, 24),
        initializer_range=0.02,
        use_bsq=True,
        bsq_dim=64,
        bsq_hidden_dim=8192,
        bsq_skip_final_layernorm=True,
        vistok_pred=False,
        vistok_pred_layernorm=False,
        vistok_pred_transformer_head=False,
        **kwargs,
    ):
        super().__init__(
            depth=depth,
            hidden_size=hidden_size,
            hidden_act=hidden_act,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            in_channels=in_channels,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
            out_hidden_size=out_hidden_size,
            num_position_embeddings=num_position_embeddings,
            deepstack_visual_indexes=list(deepstack_visual_indexes),
            initializer_range=initializer_range,
            **kwargs,
        )

        self.use_bsq = use_bsq
        self.bsq_dim = bsq_dim
        self.bsq_hidden_dim = bsq_hidden_dim
        self.bsq_skip_final_layernorm = bsq_skip_final_layernorm
        self.vistok_pred = vistok_pred
        self.vistok_pred_layernorm = vistok_pred_layernorm
        self.vistok_pred_transformer_head = vistok_pred_transformer_head


__all__ = ["UniARVisionConfig"]
