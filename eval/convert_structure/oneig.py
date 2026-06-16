"""
Convert UniAR ``run_uniar.py`` output → OneIG-Benchmark image layout.

UniAR layout (input)::

    <run_dir>/
    ├── 00000/
    │   ├── metadata.json            # {"idx", "prompt", "meta": {"category", "id", ...}}
    │   └── samples/
    │       ├── 0000.png
    │       ├── 0001.png
    │       ├── 0002.png
    │       └── 0003.png

OneIG layout (output, consumed by evaluator/scripts/text/text_score.py etc.)::

    <images_dir>/
    ├── text/
    │   └── <model_name>/
    │       ├── 000.webp       # 2x2 grid of the 4 samples
    │       ├── 001.webp
    │       └── ...
    ├── anime/<model_name>/... etc.

The upstream category → folder mapping (from ``text2image.py``) is hard-coded:
Anime_Stylization→anime, Portrait→human, General_Object→object,
Text_Rendering→text, Knowledge_Reasoning→reasoning.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from PIL import Image

CATEGORY_TO_CLASS = {
    "Anime_Stylization": "anime",
    "Portrait": "human",
    "General_Object": "object",
    "Text_Rendering": "text",
    "Knowledge_Reasoning": "reasoning",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", type=Path, required=True,
                   help="UniAR inference run directory (contains <idx:05d>/ subfolders)")
    p.add_argument("--images_dir", type=Path, default=None,
                   help="OneIG images root (default: <run_dir>/images)")
    p.add_argument("--model_name", type=str, default="uniar",
                   help="Model-name folder under each category (default: uniar)")
    p.add_argument("--grid_rows", type=int, default=2, help="Grid rows (default: 2)")
    p.add_argument("--grid_cols", type=int, default=2, help="Grid cols (default: 2)")
    p.add_argument("--webp_quality", type=int, default=95, help="WebP quality 1-100")
    p.add_argument("--num_workers", type=int, default=8, help="Parallel workers")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .webp")
    return p.parse_args()


def _load_metadata(idx_dir: Path) -> dict | None:
    meta_path = idx_dir / "metadata.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pack_grid(sample_paths: list[Path], rows: int, cols: int) -> Image.Image:
    """Paste samples into a rows×cols grid (row-major)."""
    images = [Image.open(p).convert("RGB") for p in sample_paths]
    w, h = images[0].size
    canvas = Image.new("RGB", (cols * w, rows * h))
    for i, img in enumerate(images):
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        r, c = divmod(i, cols)
        canvas.paste(img, (c * w, r * h))
    return canvas


def _process_one(
    idx_dir: Path, images_dir: Path, model_name: str,
    rows: int, cols: int, webp_quality: int, overwrite: bool,
) -> tuple[str, str | None]:
    """Returns (status, error). status ∈ {"written", "skipped", "missing", "error"}."""
    meta = _load_metadata(idx_dir)
    if not meta:
        return ("error", f"no metadata.json in {idx_dir}")
    inner = meta.get("meta", {})
    category = inner.get("category")
    prompt_id = inner.get("id")
    if not category or prompt_id is None:
        return ("error", f"meta missing category/id in {idx_dir} — got {inner!r}")
    short = CATEGORY_TO_CLASS.get(category)
    if not short:
        return ("error", f"unknown category {category!r} in {idx_dir}")

    n_needed = rows * cols
    sample_dir = idx_dir / "samples"
    if not sample_dir.exists():
        return ("missing", f"no samples/ in {idx_dir}")
    samples = sorted(sample_dir.glob("*.png"))
    if len(samples) < n_needed:
        return ("missing", f"{idx_dir}: need {n_needed} samples, have {len(samples)}")
    samples = samples[:n_needed]

    out_path = images_dir / short / model_name / f"{prompt_id}.webp"
    if out_path.exists() and not overwrite:
        return ("skipped", None)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        grid = _pack_grid(samples, rows, cols)
        grid.save(out_path, "WEBP", quality=webp_quality)
    except Exception as e:
        return ("error", f"{idx_dir}: {e}")
    return ("written", None)


def _iter_idx_dirs(run_dir: Path) -> Iterable[Path]:
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and child.name.isdigit():
            yield child


def main() -> int:
    args = _parse_args()
    if not args.run_dir.exists():
        print(f"[finalize] ERROR: run_dir not found: {args.run_dir}", file=sys.stderr)
        return 1
    images_dir = args.images_dir or (args.run_dir / "images")
    images_dir.mkdir(parents=True, exist_ok=True)

    idx_dirs = list(_iter_idx_dirs(args.run_dir))
    if not idx_dirs:
        print(f"[finalize] ERROR: no <idx>/ subfolders in {args.run_dir}", file=sys.stderr)
        return 1

    counts = {"written": 0, "skipped": 0, "missing": 0, "error": 0}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {
            ex.submit(
                _process_one, d, images_dir, args.model_name,
                args.grid_rows, args.grid_cols, args.webp_quality, args.overwrite,
            ): d
            for d in idx_dirs
        }
        for f in as_completed(futs):
            status, err = f.result()
            counts[status] += 1
            if err:
                errors.append(err)

    print(f"[finalize] {counts['written']} written, {counts['skipped']} skipped, "
          f"{counts['missing']} missing samples, {counts['error']} errors "
          f"→ {images_dir}")
    if errors:
        shown = errors[:10]
        for e in shown:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  (+ {len(errors) - 10} more)", file=sys.stderr)
    return 0 if counts["error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
