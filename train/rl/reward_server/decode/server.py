"""FastAPI server for BSQ codes decoding.

Exposes ``/decode`` and ``/decode/batch`` endpoints that accept base64-encoded
BSQ code tensors and return decoded images as base64-encoded PNGs.

Usage::

    python server.py \
        --sd3-transformer-path /path/to/sd3_transformer \
        --sd3-path /path/to/sd3 \
        --image-tokenizer-path /path/to/bsq_encoder \
        --port 8000
"""

import argparse
import asyncio
import base64
import io
import os
import pickle
import threading
import time
import traceback
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from decoder import BSQDecoder


def _pil_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class DecodeRequest(BaseModel):
    tensor_data: str
    completion_id: str = None
    image_width: int = 512
    image_height: int = 512
    num_inference_steps: int = 28
    cfg_scale: float = 1.5
    save_path: Optional[str] = None


class BatchDecodeRequest(BaseModel):
    requests: List[DecodeRequest]


class DecodeResponse(BaseModel):
    success: bool
    completion_id: str = None
    image_data: str = None
    image_path: Optional[str] = None
    error_message: str = None
    processing_time: float = None


class BatchDecodeResponse(BaseModel):
    results: List[DecodeResponse]
    total_time: float


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

_decoder: BSQDecoder = None
_gpu_lock = threading.Lock()


def _decode_one(req: DecodeRequest) -> DecodeResponse:
    """Run a single decode under the GPU lock (called from a worker thread)."""
    t0 = time.monotonic()
    try:
        tensor = pickle.loads(base64.b64decode(req.tensor_data))
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("payload is not a torch.Tensor")
        with _gpu_lock:
            out_img = _decoder.decode_bsq_codes(
                tensor,
                num_inference_steps=req.num_inference_steps,
                image_width=req.image_width,
                image_height=req.image_height,
                cfg_scale=req.cfg_scale,
            )
        image_path = None
        if req.save_path:
            out_img.save(req.save_path)
            image_path = req.save_path
        return DecodeResponse(
            success=True,
            completion_id=req.completion_id,
            image_data=_pil_to_base64(out_img),
            image_path=image_path,
            processing_time=time.monotonic() - t0,
        )
    except Exception as e:
        return DecodeResponse(
            success=False,
            completion_id=req.completion_id,
            error_message=f"{e}\n{traceback.format_exc()}",
            processing_time=time.monotonic() - t0,
        )


def create_app(config: dict) -> FastAPI:
    app = FastAPI(title="BSQ Decoder API", version="1.0.0")

    @app.on_event("startup")
    async def _startup():
        global _decoder
        device = "cuda"
        _decoder = BSQDecoder(
            image_tokenizer_path=config["image_tokenizer_path"],
            sd3_transformer_path=config["sd3_transformer_path"],
            sd3_path=config["sd3_path"],
            image_width=config["image_width"],
            image_height=config["image_height"],
            num_inference_steps=config["num_inference_steps"],
            cfg_scale=config["cfg_scale"],
            is_gt=config.get("is_gt", False),
            super_resolution=config.get("super_resolution", False),
            upscale_factor=config.get("upscale_factor", 1),
            inference_skip_final_layernorm=config.get("inference_skip_final_layernorm", False),
            device=device,
        )

    @app.get("/health")
    async def health():
        return {"status": "healthy", "decoder_loaded": _decoder is not None}

    @app.post("/decode", response_model=DecodeResponse)
    async def decode(request: DecodeRequest):
        if _decoder is None:
            raise HTTPException(status_code=500, detail="decoder not initialized")
        return await asyncio.to_thread(_decode_one, request)

    @app.post("/decode/batch", response_model=BatchDecodeResponse)
    async def decode_batch(request: BatchDecodeRequest):
        if _decoder is None:
            raise HTTPException(status_code=500, detail="decoder not initialized")
        t0 = time.monotonic()
        results = []
        for req in request.requests:
            result = await asyncio.to_thread(_decode_one, req)
            results.append(result)
        return BatchDecodeResponse(
            results=results,
            total_time=time.monotonic() - t0,
        )

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="BSQ Decoder API Server")
    p.add_argument("--sd3-transformer-path", required=True)
    p.add_argument("--sd3-path", required=True)
    p.add_argument("--image-tokenizer-path", required=True)
    p.add_argument("--image-width", type=int, default=512)
    p.add_argument("--image-height", type=int, default=512)
    p.add_argument("--num-inference-steps", type=int, default=28)
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--is-gt", action="store_true")
    p.add_argument("--super-resolution", action="store_true")
    p.add_argument("--upscale-factor", type=int, default=1)
    p.add_argument("--inference-skip-final-layernorm", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = {k: v for k, v in vars(args).items() if k not in ("host", "port", "workers")}
    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
