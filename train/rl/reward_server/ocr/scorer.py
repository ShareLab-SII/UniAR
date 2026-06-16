"""OCR scorer — PaddleOCR based edit-distance reward.

Supports both PaddleOCR 2.x (.ocr) and 3.x (.predict) APIs.
"""

import os
import re

import numpy as np
from PIL import Image
from Levenshtein import distance
from typing import List, Optional, Union


def _create_paddle_ocr(use_gpu: bool, model_dir: Optional[str]):
    """Create a PaddleOCR instance, handling API differences between 2.x and 3.x."""
    from paddleocr import PaddleOCR
    import paddleocr
    major = int(paddleocr.__version__.split(".")[0])

    if major >= 3:
        kwargs = {"lang": "en", "use_doc_orientation_classify": False,
                  "use_doc_unwarping": False, "use_textline_orientation": False}
        if not use_gpu:
            kwargs["device"] = "cpu"
        return PaddleOCR(**kwargs), "v3"
    else:
        kwargs = {"lang": "en", "use_gpu": use_gpu,
                  "use_angle_cls": False, "show_log": False}
        if model_dir is not None:
            for sub in ["det/en", "rec/en", "cls"]:
                os.makedirs(os.path.join(model_dir, sub), exist_ok=True)
            kwargs["det_model_dir"] = os.path.join(model_dir, "det", "en")
            kwargs["rec_model_dir"] = os.path.join(model_dir, "rec", "en")
            kwargs["cls_model_dir"] = os.path.join(model_dir, "cls")
        return PaddleOCR(**kwargs), "v2"


def _run_ocr(ocr, img, version):
    """Run OCR and return recognized text string."""
    if version == "v3":
        results = list(ocr.predict(img))
        texts = []
        for r in results:
            if isinstance(r, dict) and "rec_texts" in r:
                for txt, score in zip(r["rec_texts"], r["rec_scores"]):
                    if score > 0:
                        texts.append(txt)
        return "".join(texts)
    else:
        result = ocr.ocr(img)
        if result and result[0]:
            return "".join(
                res[1][0] if res[1][1] > 0 else "" for res in result[0]
            )
        return ""


class OcrScorer:
    def __init__(self, use_gpu: bool = True, model_dir: Optional[str] = None):
        self.ocr, self.version = _create_paddle_ocr(use_gpu, model_dir)

    def __call__(
        self,
        images: Union[List[Image.Image], List[np.ndarray]],
        prompts: List[str],
    ) -> list:
        def extract_quoted_texts(prompt: str) -> str:
            quoted = re.findall(r'"([^"]*)"', prompt)
            return " ".join(quoted) if quoted else ""

        prompts = [extract_quoted_texts(p) for p in prompts]
        assert len(images) == len(prompts)

        rewards = []
        for img, prompt in zip(images, prompts):
            if not prompt:
                rewards.append(0.0)
                continue
            if isinstance(img, Image.Image):
                img = np.array(img)
            try:
                recognized = _run_ocr(self.ocr, img, self.version)
                recognized = recognized.replace(" ", "").lower()
                prompt = prompt.replace(" ", "").lower()
                dist = distance(recognized, prompt)
                dist = min(dist, len(prompt))
            except Exception as e:
                print(f"OCR processing failed: {e}")
                dist = len(prompt)
            rewards.append(1 - dist / len(prompt))
        return rewards


if __name__ == "__main__":
    example_path = os.environ.get("OCR_EXAMPLE_IMAGE", "/path/to/example.png")
    example_image = Image.open(example_path)
    example_prompt = 'New York Skyline with "Hello World" written with fireworks on the sky'
    scorer = OcrScorer(use_gpu=False, model_dir=os.environ.get("OCR_MODEL_DIR"))
    reward = scorer([example_image], [example_prompt])
    print(f"OCR Reward: {reward}")
