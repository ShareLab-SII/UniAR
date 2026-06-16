"""
Convert UniAR ``run_uniar.py`` output → X-Omni TextBench sample layout.

UniAR layout (input)::

    <run_dir>/
    ├── 00000/
    │   ├── metadata.json            # {"idx", "prompt", "meta": {"prompt_id", ...}}
    │   └── samples/
    │       ├── 0000.png
    │       ├── 0001.png
    │       ├── 0002.png
    │       └── 0003.png

TextBench layout (output, consumed by evaluator/evaluate_text_reward.py)::

    <samples_dir>/
    ├── 0_0.png       # <prompt_id>_<k>.png, flat directory
    ├── 0_1.png
    ├── 0_2.png
    ├── 0_3.png
    ├── 1_0.png
    └── ...

No grid packing — each sample is a separate PNG. The evaluator parses the
leading ``<prompt_id>_`` from each filename and looks up the GT text list.
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
    p.add_argument("--expected_prompt_count", type=int, default=160,
                   help="Sanity-check: expected number of prompts (EN default 160)")
    p.add_argument("--images_per_prompt", type=int, default=4,
                   help="Sanity-check: expected samples per prompt (default 4)")
    p.add_argument("--no_check", action="store_true",
                   help="Disable prompt-count / image-count sanity checks")
    return p.parse_args()


def _load_metadata(idx_dir: Path) -> dict | None:
    meta_path = idx_dir / "metadata.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _process_one(
    idx_dir: Path, samples_dir: Path, overwrite: bool,
) -> tuple[str, str | None, int | None]:
    """Returns (status, error, prompt_id). status ∈ {"written", "skipped", "missing", "error"}."""
    meta = _load_metadata(idx_dir)
    if not meta:
        return ("error", f"no metadata.json in {idx_dir}", None)
    inner = meta.get("meta", {})
    prompt_id = inner.get("prompt_id")
    if prompt_id is None:
        return ("error", f"meta missing prompt_id in {idx_dir} — got {inner!r}", None)

    sample_dir = idx_dir / "samples"
    if not sample_dir.exists():
        return ("missing", f"no samples/ in {idx_dir}", prompt_id)
    samples = sorted(sample_dir.glob("*.png"))
    if not samples:
        return ("missing", f"no *.png under {sample_dir}", prompt_id)

    written = skipped = 0
    try:
        for k, src in enumerate(samples):
            dest = samples_dir / f"{prompt_id}_{k}.png"
            if dest.exists() and not overwrite:
                skipped += 1
                continue
            shutil.copy2(src, dest)
            written += 1
    except Exception as e:
        return ("error", f"{idx_dir}: {e}", prompt_id)

    if written == 0 and skipped > 0:
        return ("skipped", None, prompt_id)
    return ("written", None, prompt_id)


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
    seen_prompt_ids: set[int] = set()
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {
            ex.submit(_process_one, d, samples_dir, args.overwrite): d
            for d in idx_dirs
        }
        for f in as_completed(futs):
            status, err, pid = f.result()
            counts[status] += 1
            if err:
                errors.append(err)
            if pid is not None and status in ("written", "skipped"):
                seen_prompt_ids.add(pid)

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
        # Sanity checks: prompt_id coverage + per-prompt file count on disk.
        n_prompts = len(seen_prompt_ids)
        if n_prompts != args.expected_prompt_count:
            missing = sorted(set(range(args.expected_prompt_count)) - seen_prompt_ids)
            extra = sorted(seen_prompt_ids - set(range(args.expected_prompt_count)))
            print(f"[finalize] WARN: got {n_prompts} prompt_ids, expected "
                  f"{args.expected_prompt_count}", file=sys.stderr)
            if missing:
                print(f"  missing prompt_ids: {missing[:20]}"
                      + (f" (+ {len(missing) - 20} more)" if len(missing) > 20 else ""),
                      file=sys.stderr)
            if extra:
                print(f"  extra prompt_ids: {extra[:20]}", file=sys.stderr)
        # Check per-prompt file count on disk.
        bad = []
        for pid in sorted(seen_prompt_ids):
            n = len(list(samples_dir.glob(f"{pid}_*.png")))
            if n != args.images_per_prompt:
                bad.append((pid, n))
        if bad:
            print(f"[finalize] WARN: {len(bad)} prompts have ≠ {args.images_per_prompt} files:",
                  file=sys.stderr)
            for pid, n in bad[:10]:
                print(f"  prompt_id={pid}: {n} files", file=sys.stderr)

    return 0 if counts["error"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
