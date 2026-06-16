"""UniAR model configuration (Qwen3-VL config + BSQ vision + vistok_pred head)."""

from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
)

from uniar.vision_encoder.configuration_vision_encoder import UniARVisionConfig


class UniARConfig(Qwen3VLConfig):
    """
    Configuration for the UniAR autoregressive model.

    Extends ``Qwen3VLConfig`` with two UniAR-specific additions:

    - ``vision_config`` uses ``UniARVisionConfig`` (adds BSQ fields).
    - A visual-token prediction head (``output_layer_vistok``) controlled by
      ``vistok_pred`` / ``vistok_pred_layernorm`` / ``vistok_pred_use_mlp``.
      When ``visual_transformer_decoder=True``, a small transformer stack is
      inserted before the head for image-generation decoding.
    """

    model_type = "uniar"
    sub_configs = {"vision_config": UniARVisionConfig, "text_config": Qwen3VLTextConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=False,
        vistok_pred=False,
        vistok_pred_layernorm=False,
        vistok_pred_use_mlp=False,
        visual_transformer_decoder=False,
        visual_transformer_decoder_depth=4,
        **kwargs,
    ):
        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vistok_pred = vistok_pred
        self.vistok_pred_layernorm = vistok_pred_layernorm
        self.vistok_pred_use_mlp = vistok_pred_use_mlp
        self.visual_transformer_decoder = visual_transformer_decoder
        self.visual_transformer_decoder_depth = visual_transformer_decoder_depth


__all__ = ["UniARConfig"]
