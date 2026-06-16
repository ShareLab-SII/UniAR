"""
Emit per-prompt ``metadata.jsonl`` files that the upstream GenEval evaluator
(``evaluator/evaluate_images.py``) expects.

``run_uniar.py`` writes a unified ``metadata.json`` under each
``<run>/<idx:05d>/`` directory containing ``{idx, prompt, meta: {...}}``. This
script walks the run directory and writes a one-line ``metadata.jsonl`` next to
each ``metadata.json`` containing only the upstream fields
(``tag``, ``include``, ``prompt``, optional ``exclude``).

Example::

    python eval/geneval/finalize.py --run_dir eval/runs/geneval_smoke
"""

import argparse
import json
from pathlib import Path


_REQUIRED = {"tag", "include", "prompt"}
_OPTIONAL = {"exclude"}


def _to_eval_record(meta: dict) -> dict:
    missing = _REQUIRED - meta.keys()
    if missing:
        raise ValueError(f"meta missing required GenEval fields {missing}: {meta}")
    # When the rewrite-augmented long jsonl is used, ``meta['prompt']`` is the
    # long rewritten prompt (the one we actually generated with) and
    # ``meta['short_prompt']`` is the canonical upstream GenEval prompt. The
    # evaluator only uses ``prompt`` as an echo-through in its output, but
    # reports conventionally show the canonical short form, so we prefer
    # ``short_prompt`` when present.
    eval_prompt = meta.get("short_prompt", meta["prompt"])
    rec = {"tag": meta["tag"], "include": meta["include"], "prompt": eval_prompt}
    for k in _OPTIONAL:
        if k in meta:
            rec[k] = meta[k]
    return rec


def main():
    parser = argparse.ArgumentParser(description="Emit per-prompt metadata.jsonl for GenEval evaluator")
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Directory containing <idx:05d>/metadata.json written by run_uniar.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-emit even if metadata.jsonl already exists.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir not found: {run_dir}")

    n_done = n_skipped = 0
    for idx_dir in sorted(run_dir.iterdir()):
        if not idx_dir.is_dir() or not idx_dir.name.isdigit():
            continue
        src = idx_dir / "metadata.json"
        dst = idx_dir / "metadata.jsonl"
        if not src.exists():
            continue
        if dst.exists() and not args.overwrite:
            n_skipped += 1
            continue
        with src.open(encoding="utf-8") as f:
            payload = json.load(f)
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"{src} is missing a dict 'meta' field; was run_uniar.py fed the unified jsonl?")
        eval_rec = _to_eval_record(meta)
        with dst.open("w", encoding="utf-8") as f:
            f.write(json.dumps(eval_rec, ensure_ascii=False) + "\n")
        n_done += 1

    print(f"wrote {n_done} metadata.jsonl files; skipped {n_skipped} existing (pass --overwrite to force).")


if __name__ == "__main__":
    main()
