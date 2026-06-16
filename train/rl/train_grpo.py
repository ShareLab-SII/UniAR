import os
from dataclasses import dataclass, field
from typing import Optional

os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_LOG_MODEL", "false")
os.environ.setdefault("WANDB_WATCH", "false")
os.environ.setdefault("WANDB__DISABLE_STATS", "true")

import torch

from trl import (
    GRPOConfig,
    UniARGRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
)

from transformers import AutoProcessor

from uniar import UniARForConditionalGeneration

from dataset import RLDataset
from reward_funcs import hpsv2_reward, ocr_reward, geneval_reward, unified_reward


@dataclass
class GRPOScriptArguments(ScriptArguments):
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to training data (jsonl or yaml)"},
    )
    prompt_key: Optional[str] = field(
        default="instruction",
        metadata={"help": "Key in the JSONL records that holds the text prompt"},
    )
    image_width: Optional[int] = field(default=512, metadata={"help": "Image width"})
    image_height: Optional[int] = field(default=512, metadata={"help": "Image height"})
    reward_weights_str: Optional[str] = field(
        default=None, metadata={"help": "Reward weights, e.g. '[1.0, 1.0]'"},
    )
    reward_function_names_str: Optional[str] = field(
        default=None,
        metadata={"help": "Reward function names, e.g. '[unified_reward, hpsv2_reward]'"},
    )


def main(script_args, training_args, model_args):
    device = "cuda"

    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        torch_dtype="auto",
        output_loading_info=False,
        ignore_mismatched_sizes=True,
        _attn_implementation=model_args.attn_implementation,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    model = UniARForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, **training_args.model_init_kwargs,
    )
    model = model.to(torch.bfloat16).to(device)

    model.image_height = script_args.image_height
    model.image_width = script_args.image_width

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, padding_side='left')

    model.requires_grad_(True)
    model.visual.eval()
    model.visual.requires_grad_(False)

    downsample_factor = model.config.vision_config.patch_size * model.config.vision_config.spatial_merge_size
    train_dataset = RLDataset(
        data_path=script_args.data_path,
        prompt_key=script_args.prompt_key,
        processor=processor,
        ar_height=script_args.image_height,
        ar_width=script_args.image_width,
        downsample_factor=downsample_factor,
    )

    trainer = UniARGRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=training_args.reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=None,
    )

    resume = training_args.resume_from_checkpoint
    if isinstance(resume, str) and resume.lower() in ("false", "none", ""):
        resume = False
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    os.makedirs(training_args.output_dir, exist_ok=True)
    torch.save({
        "script_args": script_args.__dict__,
        "training_args": training_args.__dict__,
        "model_args": model_args.__dict__,
    }, os.path.join(training_args.output_dir, "args.pt"))

    training_args.reward_weights = eval(script_args.reward_weights_str)
    training_args.reward_funcs = eval(script_args.reward_function_names_str)

    main(script_args, training_args, model_args)
