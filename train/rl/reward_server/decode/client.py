"""Async client for multi-GPU BSQ decode servers.

Distributes decode requests across multiple GPU-backed server instances
using round-robin scheduling, with automatic health checking and failover.

Usage::

    client = DecoderClient(["http://localhost:8000", "http://localhost:8001"])
    results = await client.decode_batch(tensor_list, completion_ids, image_width=512, image_height=512)
"""

import asyncio
import base64
import io
import os
import pickle
import time
from dataclasses import dataclass
from typing import List, Optional

import aiohttp
import torch
from PIL import Image


@dataclass
class DecodeResult:
    success: bool
    completion_id: str = None
    image_data: str = None
    image_path: str = None
    error_message: str = None
    processing_time: float = None
    gpu_id: int = None
    original_index: int = None


def base64_to_pil(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data)))


class DecoderClient:
    """Multi-GPU decode client with round-robin load balancing."""

    def __init__(self, gpu_urls: List[str], timeout: int = 3000):
        self.gpu_urls = gpu_urls
        self.timeout = timeout
        self.gpu_stats = {url: {"requests": 0, "errors": 0, "last_used": 0.0} for url in gpu_urls}
        self._rr_index = 0

    async def health_check_all(self) -> dict:
        status = {}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            tasks = [self._check_gpu(session, url) for url in self.gpu_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for url, result in zip(self.gpu_urls, results):
                healthy = result if isinstance(result, bool) else False
                status[url] = healthy
                if healthy:
                    self.gpu_stats[url]["errors"] = 0
                else:
                    self.gpu_stats[url]["errors"] += 1
        return status

    async def _check_gpu(self, session, url) -> bool:
        try:
            async with session.get(f"{url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    def _select_gpu(self, healthy_urls: List[str]) -> str:
        url = healthy_urls[self._rr_index % len(healthy_urls)]
        self._rr_index += 1
        return url

    async def decode_batch(
        self,
        tensor_list: List[torch.Tensor],
        completion_ids: List[str] = None,
        image_width: int = 512,
        image_height: int = 512,
        num_inference_steps: int = None,
        cfg_scale: float = None,
        save_paths: List[str] = None,
    ) -> List[DecodeResult]:
        n = len(tensor_list)
        if completion_ids is None:
            completion_ids = [f"batch_{i}" for i in range(n)]

        health = await self.health_check_all()
        healthy = [u for u, ok in health.items() if ok]
        if not healthy:
            return [DecodeResult(success=False, completion_id=cid, error_message="no healthy GPU")
                    for cid in completion_ids]

        # Assign items to GPUs round-robin
        groups: dict[str, dict] = {}
        for i in range(n):
            url = self._select_gpu(healthy)
            grp = groups.setdefault(url, {"tensors": [], "cids": [], "indices": [], "saves": []})
            grp["tensors"].append(tensor_list[i])
            grp["cids"].append(completion_ids[i])
            grp["indices"].append(i)
            grp["saves"].append(save_paths[i] if save_paths else None)

        tasks = [
            self._send_group(url, grp, image_width, image_height, num_inference_steps, cfg_scale)
            for url, grp in groups.items()
        ]
        group_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: list[Optional[DecodeResult]] = [None] * n
        for url, result in zip(groups.keys(), group_results):
            grp = groups[url]
            if isinstance(result, Exception):
                for i, cid in zip(grp["indices"], grp["cids"]):
                    all_results[i] = DecodeResult(
                        success=False, completion_id=cid,
                        error_message=str(result), original_index=i,
                    )
            else:
                for r in result:
                    if r.original_index is not None:
                        all_results[r.original_index] = r

        for i in range(n):
            if all_results[i] is None:
                all_results[i] = DecodeResult(
                    success=False, completion_id=completion_ids[i],
                    error_message="result not returned", original_index=i,
                )
        return all_results

    async def _send_group(self, url, grp, image_width, image_height, num_inference_steps, cfg_scale):
        requests = []
        for tensor, cid, save in zip(grp["tensors"], grp["cids"], grp["saves"]):
            req = {
                "tensor_data": base64.b64encode(pickle.dumps(tensor)).decode("utf-8"),
                "completion_id": cid,
                "image_width": image_width,
                "image_height": image_height,
                "save_path": save,
            }
            if num_inference_steps is not None:
                req["num_inference_steps"] = num_inference_steps
            if cfg_scale is not None:
                req["cfg_scale"] = cfg_scale
            requests.append(req)

        gpu_id = self.gpu_urls.index(url) if url in self.gpu_urls else None
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(f"{url}/decode/batch", json={"requests": requests}) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {text}")
                data = await resp.json()

        results = []
        for i, rd in enumerate(data["results"]):
            r = DecodeResult(
                success=rd.get("success", False),
                completion_id=rd.get("completion_id"),
                image_data=rd.get("image_data"),
                image_path=rd.get("image_path"),
                error_message=rd.get("error_message"),
                processing_time=rd.get("processing_time"),
                gpu_id=gpu_id,
                original_index=grp["indices"][i],
            )
            results.append(r)

        self.gpu_stats[url]["requests"] += len(grp["tensors"])
        self.gpu_stats[url]["last_used"] = time.time()
        return results

    def get_stats(self) -> dict:
        return {
            "gpu_stats": self.gpu_stats,
            "total_requests": sum(s["requests"] for s in self.gpu_stats.values()),
            "total_errors": sum(s["errors"] for s in self.gpu_stats.values()),
        }


if __name__ == "__main__":
    server_ip = os.environ.get("DECODER_SERVER_IP", "localhost")
    num_gpus = int(os.environ.get("NUM_GPUS", 4))

    async def demo():
        gpu_urls = [f"http://{server_ip}:{8000 + i}" for i in range(num_gpus)]
        client = DecoderClient(gpu_urls)

        health = await client.health_check_all()
        print(f"healthy: {sum(health.values())}/{len(health)}")

        bsq_pt = os.environ.get("BSQ_CODES_PT")
        if not bsq_pt:
            print("Set BSQ_CODES_PT to run a decode demo.")
            return

        codes = torch.load(bsq_pt)
        tensors = [codes[i] for i in range(min(num_gpus, len(codes)))]
        results = await client.decode_batch(tensors, image_width=512, image_height=512)
        for r in results:
            status = "ok" if r.success else f"FAIL: {r.error_message}"
            print(f"  [{r.completion_id}] gpu={r.gpu_id} {status}")

    asyncio.run(demo())
