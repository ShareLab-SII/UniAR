"""
Convert UniAR ``run_uniar.py`` (edit-mode) output → ImgEdit Basic-Bench layout.

UniAR layout (input)::

    <run_dir>/
    ├── 00000/
    │   ├── metadata.json
    │   │   # {"idx", "prompt", "meta": {"metadata": {"original_id", ...}, ...}}
    │   └── samples/
    │       └── 0000.png

ImgEdit Basic-Bench layout (output, consumed by evaluator/basic_bench.py)::

    <samples_dir>/
    ├── 1082.png    # <original_id>.png, flat directory
    ├── 1068.png
    └── ...

ImgEdit Basic convention: one sample per prompt (``samples_per_prompt=1``).
Only the first sample under each idx is copied; additional samples are
skipped with a warning. Sanity-checks by default that all 737 original_ids
from ``imgedit_single_turn.jsonl`` are present.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", type=Path, required=True,
                   help="UniAR inference run directory (contains <idx:05d>/ subfolders)")
    p.add_argument("--samples_dir", type=Path, default=None,
                   help="Flat output dir for evaluator (default: <run_dir>/samples_flat)")
    p.add_argument("--num_workers", type=int, default=16, help="Parallel workers")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .png")
    p.add_argument("--expected_prompt_count", type=int, default=737,
                   help="Sanity-check: expected number of prompts (ImgEdit Basic default 737)")
    p.add_argument("--no_check", action="store_true",
                   help="Disable prompt-count coverage sanity check")
    return p.parse_args()


def _load_metadata(idx_dir: Path) -> dict | None:
    meta_path = idx_dir / "metadata.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _process_one(
    idx_dir: Path, samples_dir: Path, overwrite: bool,
) -> tuple[str, str | None, str | None]:
    """Returns (status, error, original_id). status ∈ {"written", "skipped", "missing", "error"}."""
    meta = _load_metadata(idx_dir)
    if not meta:
        return ("error", f"no metadata.json in {idx_dir}", None)
    # Unified schema: outer meta["meta"] is the upstream row which has "metadata"
    inner = meta.get("meta", {})
    upstream_meta = inner.get("metadata", {}) if isinstance(inner, dict) else {}
    original_id = upstream_meta.get("original_id")
    if original_id is None:
        return ("error",
                f"meta missing metadata.original_id in {idx_dir}", None)

    sample_dir = idx_dir / "samples"
    if not sample_dir.exists():
        return ("missing", f"no samples/ in {idx_dir}", str(original_id))
    samples = sorted(sample_dir.glob("*.png"))
    if not samples:
        return ("missing", f"no *.png under {sample_dir}", str(original_id))

    # ImgEdit Basic convention: one output image per prompt.
    # If samples_per_prompt > 1, only the first is copied (deterministic choice).
    src = samples[0]
    dest = samples_dir / f"{original_id}.png"
    if dest.exists() and not overwrite:
        return ("skipped", None, str(original_id))
    try:
        shutil.copy2(src, dest)
    except Exception as e:
        return ("error", f"{idx_dir}: {e}", str(original_id))
    return ("written", None, str(original_id))


def _iter_idx_dirs(run_dir: Path) -> Iterable[Path]:
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and child.name.isdigit():
            yield child


def main() -> int:
    args = _parse_args()
    if not args.run_dir.exists():
        print(f"[finalize] ERROR: run_dir not found: {args.run_dir}", file=sys.stderr)
        return 1
    samples_dir = args.samples_dir or (args.run_dir / "samples_flat")
    samples_dir.mkdir(parents=True, exist_ok=True)

    idx_dirs = list(_iter_idx_dirs(args.run_dir))
    if not idx_dirs:
        print(f"[finalize] ERROR: no <idx>/ subfolders in {args.run_dir}", file=sys.stderr)
        return 1

    counts = {"written": 0, "skipped": 0, "missing": 0, "error": 0}
    errors: list[str] = []
    seen_ids: set[str] = set()
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {
            ex.submit(_process_one, d, samples_dir, args.overwrite): d
            for d in idx_dirs
        }
        for f in as_completed(futs):
            status, err, oid = f.result()
            counts[status] += 1
            if err:
                errors.append(err)
            if oid is not None and status in ("written", "skipped"):
                seen_ids.add(oid)

    print(f"[finalize] {counts['written']} written, {counts['skipped']} skipped, "
          f"{counts['missing']} missing samples, {counts['error']} errors "
          f"→ {samples_dir}")
    if errors:
        shown = errors[:10]
        for e in shown:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  (+ {len(errors) - 10} more)", file=sys.stderr)

    if not args.no_check:
        n = len(seen_ids)
        if n != args.expected_prompt_count:
            print(f"[finalize] WARN: got {n} unique original_ids, expected "
                  f"{args.expected_prompt_count}", file=sys.stderr)
        else:
            print(f"[finalize] ✓ {n}/{args.expected_prompt_count} original_ids covered")

    return 0 if counts["error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
