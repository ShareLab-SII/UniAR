"""OCR evaluator — PaddleOCR word-recall scoring.

Requires:
    OCR_MODEL_DIR  (optional) directory for PaddleOCR model weights
"""

import os
import sys
import time

from scorer import OcrScorer


def load_ocr():
    def timed(fn):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = fn(*args, **kwargs)
            print(f"Function {fn.__name__!r} executed in {time.time() - t0:.3f}s", file=sys.stderr)
            return result
        return wrapper

    @timed
    def load_models():
        model_dir = os.environ.get("OCR_MODEL_DIR")
        ocr_scorer = OcrScorer(use_gpu=False, model_dir=model_dir)
        return ocr_scorer

    ocr_model = load_models()

    def compute_ocr(prompts, images):
        rewards = ocr_model(images, prompts)
        native_rewards = []
        for r in rewards:
            if hasattr(r, "item") and callable(r.item):
                r = r.item()
            native_rewards.append(float(r))
        return native_rewards

    return compute_ocr
