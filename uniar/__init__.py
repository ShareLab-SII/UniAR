"""
UniAR: Unified Autoregressive Multimodal Model.

Qwen3-VL backbone + BSQ vision encoder + visual-token prediction head,
packaged for image understanding and image generation in a single model.
"""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from uniar.configuration_uniar import UniARConfig
from uniar.modeling_uniar import UniARForConditionalGeneration, UniARModel
from uniar.vision_encoder.configuration_vision_encoder import UniARVisionConfig
from uniar.vision_encoder.modeling_vision_encoder import UniARVisionModel

AutoConfig.register(UniARConfig.model_type, UniARConfig, exist_ok=True)
AutoConfig.register(UniARVisionConfig.model_type, UniARVisionConfig, exist_ok=True)
AutoModel.register(UniARConfig, UniARModel, exist_ok=True)
AutoModel.register(UniARVisionConfig, UniARVisionModel, exist_ok=True)
AutoModelForCausalLM.register(UniARConfig, UniARForConditionalGeneration, exist_ok=True)

CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}"
    "{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}{% endif %}"
    "{% endfor %}<|im_end|>\n{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

def __getattr__(name):
    if name == "UniARVisualDecoder":
        from uniar.modeling_vision_decoder import UniARVisualDecoder
        return UniARVisualDecoder
    raise AttributeError(f"module 'uniar' has no attribute {name}")


__all__ = [
    "CHAT_TEMPLATE",
    "UniARConfig",
    "UniARForConditionalGeneration",
    "UniARModel",
    "UniARVisualDecoder",
    "UniARVisionConfig",
    "UniARVisionModel",
]
