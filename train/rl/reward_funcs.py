"""Reward functions for GRPO RL training.

Each function takes ``(images: List[PIL.Image], **kwargs)`` and returns a list
of reward scores (``None`` for samples where the reward does not apply, e.g.
a geneval reward on a non-geneval prompt).

Supported rewards:
- ``hpsv2_reward``         : human preference score v2 (applies to all samples)
- ``geneval_reward``       : GenEval task check (task == 'geneval' only)
- ``ocr_reward``           : OCR word-recall (task == 'ocr' only)
- ``unified_reward``       : generic VLM judge via vLLM (applies to all samples)

Each reward hits an HTTP endpoint configured via its ``*_API_ADDRESS`` env var
(or ``api_address`` kwarg). See ``reward_server/<name>/README.md`` for how to
start each server.
"""

import io
import logging
import os
import pickle
import time

import requests

from reward_server.unified_reward.client import evaluate_batch

logger = logging.getLogger(__name__)


def _post_with_retry(url, data, timeout=600, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, data=data, timeout=timeout)
            resp.raise_for_status()
            return pickle.loads(resp.content)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                logger.warning(f"[reward] request failed (attempt {attempt+1}/{max_retries}): {e}, retrying in {wait}s")
                time.sleep(wait)
            else:
                raise


def hpsv2_reward(images, **kwargs):
    """HPS v2.1 human-preference reward. Applies to all samples."""
    api_address = kwargs.get("api_address", os.environ.get("HPSV2_API_ADDRESS"))
    jpeg_data = []
    for image in images:
        buffer = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG")
        jpeg_data.append(buffer.getvalue())

    raw_prompts = kwargs.get("raw_prompt")
    payload = {"images": jpeg_data, "prompts": raw_prompts}
    result = _post_with_retry(api_address, pickle.dumps(payload))
    return result["scores"]


def geneval_reward(images, **kwargs):
    """GenEval compositional task-check reward. Only applies to task == 'geneval'."""
    api_address = kwargs.get("api_address", os.environ.get("GENEVAL_API_ADDRESS"))
    only_strict = kwargs.get("only_strict", False)

    tasks = kwargs.get("task")
    raw_prompts = kwargs.get("raw_prompt")
    metadatas = kwargs.get("metadata")

    if "geneval" not in tasks:
        return [None for _ in range(len(images))]

    geneval_images, geneval_metadatas, geneval_idx = [], [], []
    for idx, (img, t, raw_p, meta) in enumerate(zip(images, tasks, raw_prompts, metadatas)):
        if t != "geneval":
            continue
        buffer = io.BytesIO()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buffer, format="JPEG")
        geneval_images.append(buffer.getvalue())
        meta["prompt"] = raw_p
        geneval_metadatas.append(meta)
        geneval_idx.append(idx)

    data = {"images": geneval_images, "meta_datas": geneval_metadatas, "only_strict": only_strict}
    response_data = _post_with_retry(api_address, pickle.dumps(data))

    geneval_rewards = response_data["scores"]

    rewards = [None for _ in range(len(images))]
    for idx, reward in zip(geneval_idx, geneval_rewards):
        rewards[idx] = reward
    return rewards


def ocr_reward(images, **kwargs):
    """OCR word-recall reward via PaddleOCR server. Only applies to task == 'ocr'."""
    api_address = kwargs.get("api_address", os.environ.get("OCR_API_ADDRESS"))
    tasks = kwargs.get("task")
    raw_prompts = kwargs.get("raw_prompt")

    if "ocr" not in tasks:
        return [None for _ in range(len(images))]

    ocr_images, ocr_prompts, ocr_idx = [], [], []
    for idx, (img, t, raw_p) in enumerate(zip(images, tasks, raw_prompts)):
        if t != "ocr":
            continue
        buffer = io.BytesIO()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buffer, format="JPEG")
        ocr_images.append(buffer.getvalue())
        ocr_prompts.append(raw_p)
        ocr_idx.append(idx)

    payload = {"images": ocr_images, "prompts": ocr_prompts}
    response_data = _post_with_retry(api_address, pickle.dumps(payload))
    ocr_rewards = response_data["scores"]

    rewards = [None for _ in range(len(images))]
    for idx, reward in zip(ocr_idx, ocr_rewards):
        rewards[idx] = reward
    return rewards


def _safe_extract_score(text, default=0.0, min_val=1.0, max_val=5.0):
    """Extract a score from a possibly-noisy model output line."""
    import re

    try:
        score = float(text.strip())
    except (ValueError, AttributeError):
        try:
            match = re.search(r"(\d+\.?\d*)", str(text))
            if match:
                score = float(match.group(1))
            else:
                return default
        except (ValueError, AttributeError):
            return default
    if score < min_val or score > max_val:
        score = max(min_val, min(max_val, score))
    return score


def unified_reward(images, **kwargs):
    """Unified VLM-judge reward via vLLM. Alignment + coherence + style on 1–5 scale,
    combined via weights (0.4 / 0.4 / 0.2) and normalized to [0, 1]."""
    api_address = kwargs.get("api_address", os.environ.get("UNIFIED_REWARD_API_ADDRESS"))
    raw_prompts = kwargs.get("raw_prompt")

    batch_data = []
    for raw_prompt, image in zip(raw_prompts, images):
        problem = (
            "You are presented with a generated image and its associated text caption. "
            "Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n"
            "Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
            "- Alignment Score: How well the image matches the caption in terms of content.\n"
            "- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n"
            "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
            "Output your evaluation using the format below:\n\n"
            "Alignment Score (1-5): X\n"
            "Coherence Score (1-5): Y\n"
            "Style Score (1-5): Z\n\n"
            "Your task is provided as follows:\n"
            f"Text Caption: [{raw_prompt}]"
        )
        batch_data.append({"problem": problem, "images": [image]})

    results = evaluate_batch(batch_data, api_address, image_root=None, server_name="UnifiedReward")

    rewards = []
    for result in results:
        output = result["model_output"]
        alignment_score = coherence_score = style_score = 0.0
        try:
            for line in output.split("\n"):
                if "Alignment Score" in line:
                    score_text = line.split(":", 1)[1] if ":" in line else ""
                    alignment_score = _safe_extract_score(score_text, default=0.0)
                elif "Coherence Score" in line:
                    score_text = line.split(":", 1)[1] if ":" in line else ""
                    coherence_score = _safe_extract_score(score_text, default=0.0)
                elif "Style Score" in line:
                    score_text = line.split(":", 1)[1] if ":" in line else ""
                    style_score = _safe_extract_score(score_text, default=0.0)
        except Exception as e:
            logger.warning(f"[unified_reward] failed to parse model output: {e}")

        reward = (alignment_score * 0.4 + coherence_score * 0.4 + style_score * 0.2) / 5.0
        rewards.append(reward)
    return rewards
