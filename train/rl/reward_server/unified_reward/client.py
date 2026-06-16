"""Unified reward client — VLM judge via vLLM.

Sends image + prompt pairs to a vLLM-served VLM (e.g. Qwen3-VL) for evaluation.
The VLM scores images on alignment, coherence, and style (1-5 scale each).

Start the vLLM server with::

    bash run_vllm.sh
"""

import base64
import concurrent.futures
import json
import os
import time
from io import BytesIO
from multiprocessing import Manager

import requests
from PIL import Image


class VLMessageClient:
    def __init__(self, api_url):
        self.api_url = api_url
        self.session = requests.Session()

    def _encode_image(self, image):
        if isinstance(image, str):
            with Image.open(image) as img:
                img = img.convert("RGB")
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=95)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            img = image.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=95)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    def build_messages(self, item, image_root=None):
        content = []
        if image_root:
            for i in range(len(item["images"])):
                if isinstance(item["images"][i], str):
                    item["images"][i] = os.path.join(image_root, item["images"][i])

        for image_item in item["images"]:
            if isinstance(image_item, str):
                if os.path.exists(image_item):
                    b64 = self._encode_image(image_item)
                else:
                    b64 = image_item
            else:
                b64 = self._encode_image(image_item)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        content.append({"type": "text", "text": item["problem"]})
        return [{"role": "user", "content": content}]

    def process_item(self, item, image_root, output_file, total_counter, lock, server_name="UnifiedReward"):
        max_retries = 3
        result = None
        for attempt in range(1, max_retries + 1):
            try:
                messages = self.build_messages(item, image_root)
                payload = {
                    "model": server_name,
                    "messages": messages,
                    "do_sample": False,
                    "max_tokens": 4096,
                }
                response = self.session.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=payload,
                    timeout=3000,
                )
                response.raise_for_status()
                output = response.json()["choices"][0]["message"]["content"]
                with lock:
                    total_counter.value += 1
                result = {
                    "problem": item["problem"],
                    "model_output": output,
                    "success": True,
                    "idx": item.get("idx", "unknown"),
                }
                break
            except Exception as e:
                if attempt == max_retries:
                    result = {
                        "problem": item["problem"],
                        "error": str(e),
                        "attempt": attempt,
                        "success": False,
                        "idx": item.get("idx", "unknown"),
                    }
                else:
                    time.sleep(min(2 ** attempt, 10))
        return result, result.get("success", False) if result else False


def evaluate_batch(batch_data, api_url, image_root=None, server_name="UnifiedReward"):
    with Manager() as manager:
        total_counter = manager.Value("i", 0)
        lock = manager.Lock()
        total_result = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
            client = VLMessageClient(api_url)
            futures = []
            for index, item in enumerate(batch_data):
                if "idx" not in item:
                    item["idx"] = str(index)
                futures.append(
                    executor.submit(
                        client.process_item,
                        item=item,
                        image_root=image_root,
                        output_file="./results.json",
                        total_counter=total_counter,
                        lock=lock,
                        server_name=server_name,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                try:
                    result, _ = future.result()
                    total_result.append(result)
                except Exception as e:
                    print(f"task exception: {e}")

    total_result.sort(key=lambda x: int(x["idx"]))
    return total_result
