"""HPSv2 scorer — ViT-H-14 based human preference model."""

import os
import torch
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer


class HPSv2:
    def __init__(self, args):
        self.ckpt_path = args.hps_ckpt_path
        self.clip_path = args.clip_path

    @property
    def __name__(self):
        return "HPSv2"

    def load_to_device(self, device):
        self.model, self.preprocess_train, self.preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            pretrained=self.clip_path,
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"])
        for param in self.model.parameters():
            param.requires_grad = False

        self.tokenizer = get_tokenizer("ViT-H-14")
        self.model = self.model.to(device)
        self.model.eval()

    def __call__(self, prompts, images, **kwargs):
        device = next(self.model.parameters()).device
        result = []
        for prompt, image in zip(prompts, images):
            with torch.no_grad():
                image_t = self.preprocess_val(image).unsqueeze(0).to(device=device, non_blocking=True)
                text = self.tokenizer([prompt]).to(device=device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda"):
                    outputs = self.model(image_t, text)
                    image_features = outputs["image_features"]
                    text_features = outputs["text_features"]
                    logits = image_features @ text_features.T
                    hps_score = torch.diagonal(logits).cpu().numpy()
            result.append(hps_score[0])
        return result
