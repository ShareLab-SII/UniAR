"""HPSv2 evaluator — human preference scoring.

Requires:
    HPSV2_CKPT   path to HPSv2 checkpoint (.pt)
    CLIP_PATH     path to pretrained CLIP weights
"""

import os
import sys
import time

import torch

from scorer import HPSv2


def load_hpsv2():
    def timed(fn):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = fn(*args, **kwargs)
            print(f"Function {fn.__name__!r} executed in {time.time() - t0:.3f}s", file=sys.stderr)
            return result
        return wrapper

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @timed
    def load_models():
        ckpt_path = os.environ.get("HPSV2_CKPT")
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                "HPSV2_CKPT not set or path does not exist. "
                "Set it to the HPSv2 checkpoint path."
            )
        clip_path = os.environ.get("CLIP_PATH")
        if not clip_path or not os.path.exists(clip_path):
            raise FileNotFoundError(
                "CLIP_PATH not set or path does not exist. "
                "Set it to the pretrained CLIP weights path."
            )
        from types import SimpleNamespace
        hps = HPSv2(SimpleNamespace(hps_ckpt_path=ckpt_path, clip_path=clip_path))
        hps.load_to_device(DEVICE)
        return hps

    hps_model = load_models()

    @torch.no_grad()
    def compute_hpsv2(prompts, images):
        scores = hps_model(prompts, images)
        native_scores = []
        for s in scores:
            if isinstance(s, torch.Tensor):
                s = s.item()
            elif hasattr(s, "item") and callable(s.item):
                s = s.item()
            native_scores.append(float(s))
        return native_scores

    return compute_hpsv2
