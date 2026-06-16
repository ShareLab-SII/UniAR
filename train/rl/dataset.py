"""RL training dataset.

Reads either a single JSONL file or a YAML dataset-mix config. Each JSONL
record must have:

    {
      "instruction": "<text prompt>",   # key overridable via prompt_key
      "number": <int>,                  # unique id for logging
      "task": "<optional: 'geneval' | 'ocr' | ...>",
      "metadata": { ... }                # optional, reward-specific
    }

YAML config schema::

    datasets:
      - name: "mix_a"
        json_path: "/path/to/a.jsonl"
        sampled_ratio: 0.5       # fraction (0,1)  OR  1 = all  OR  integer count
      - name: "mix_b"
        json_path: "/path/to/b.jsonl"
        sampled_ratio: 1000
"""

import json
import logging
import random

import yaml
from torch.utils.data import Dataset
from transformers import AutoProcessor

from uniar import CHAT_TEMPLATE
from inference.visual_inputs import _build_visual_messages

logger = logging.getLogger(__name__)


class RLDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        processor: AutoProcessor,
        ar_height: int,
        ar_width: int,
        downsample_factor: int,
        prompt_key: str = "instruction",
    ):
        super().__init__()
        self.data_path = data_path
        self.prompt_key = prompt_key
        if data_path.endswith(".jsonl"):
            with open(data_path, "r") as f:
                self.data = [json.loads(line) for line in f]
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as f:
                config = yaml.safe_load(f)

            self.data = []
            total_samples = 0
            for dataset_config in config.get("datasets", []):
                json_path = dataset_config["json_path"]
                sampled_ratio = dataset_config["sampled_ratio"]
                name = dataset_config.get("name", "unknown")
                with open(json_path, "r") as f:
                    dataset_items = [json.loads(line) for line in f]
                total_items = len(dataset_items)
                if sampled_ratio == 1:
                    sampled_count = total_items
                    sampled_items = dataset_items
                elif 0 < sampled_ratio < 1:
                    sampled_count = min(int(total_items * sampled_ratio), total_items)
                    sampled_items = random.sample(dataset_items, sampled_count)
                else:
                    sampled_count = min(int(sampled_ratio), total_items)
                    sampled_items = random.sample(dataset_items, sampled_count)
                self.data.extend(sampled_items)
                total_samples += sampled_count
                logger.info(f"dataset '{name}': total={total_items}, sampled={sampled_count}")
            logger.info(f"total samples: {total_samples}")
        else:
            raise ValueError(f"Invalid data path: {data_path}")

        self.processor = processor
        self.ar_height = ar_height
        self.ar_width = ar_width
        self.downsample_factor = downsample_factor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return {
            "prompt": self._build_prompt(item[self.prompt_key]),
            "raw_prompt": item[self.prompt_key],
            "task": item.get("task", None),
            "metadata": item.get("metadata", None),
            "number": item["number"],
        }

    def _build_prompt(self, prompt: str) -> str:
        messages = _build_visual_messages(
            prompt, self.ar_height, self.ar_width, self.downsample_factor,
        )
        original_template = self.processor.chat_template
        self.processor.chat_template = CHAT_TEMPLATE
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.processor.chat_template = original_template
        return text + "<|vision_start|>"
