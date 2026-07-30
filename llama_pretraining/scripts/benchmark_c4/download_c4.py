#!/usr/bin/env python3
"""Prefetch C4 + t5-base into the HuggingFace cache for offline/cluster training.

Training uses streaming allenai/c4; files already in the hub cache are reused.
Validation is downloaded fully. Train shards are partial by default (full C4 is huge).

Examples:
  python download_c4.py
  python download_c4.py --hf-home /scratch/$USER/hf_cache --train-shards 32
  HF_HOME=/pscratch/$USER/hf python download_c4.py --train-shards 0   # tokenizer + val only
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--hf-home",
        type=str,
        default=os.environ.get("HF_HOME") or os.environ.get("HF_DATASETS_CACHE") or "",
        help="Cache root (sets HF_HOME). Default: existing HF_HOME or ~/.cache/huggingface",
    )
    p.add_argument(
        "--train-shards",
        type=int,
        default=16,
        help="Number of C4 English train shards to download (00000..). 0 skips train. Default 16.",
    )
    p.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip downloading t5-base tokenizer/model files.",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip C4 validation download.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Open streaming train+val briefly to verify the cache works.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
        os.makedirs(args.hf_home, exist_ok=True)
        print(f"[download_c4] HF_HOME={args.hf_home}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required (pip install huggingface_hub)", file=sys.stderr)
        sys.exit(1)

    if not args.skip_tokenizer:
        print("[download_c4] downloading t5-base (tokenizer used by training)...")
        path = snapshot_download(repo_id="t5-base", repo_type="model")
        print(f"[download_c4] t5-base -> {path}")

    allow = []
    if not args.skip_validation:
        allow.append("en/c4-validation.*.json.gz")
    if args.train_shards > 0:
        # shard ids are zero-padded to 5 digits in allenai/c4
        n = min(args.train_shards, 1024)
        for i in range(n):
            allow.append(f"en/c4-train.{i:05d}-of-*.json.gz")

    if allow:
        print(f"[download_c4] downloading allenai/c4 patterns: {allow[:3]}{'...' if len(allow) > 3 else ''}")
        path = snapshot_download(
            repo_id="allenai/c4",
            repo_type="dataset",
            allow_patterns=allow,
        )
        print(f"[download_c4] c4 -> {path}")
    else:
        print("[download_c4] no C4 shard patterns requested")

    if args.smoke:
        print("[download_c4] smoke: streaming open train/validation...")
        import datasets
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("t5-base")
        train = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
        val = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True)
        t0 = next(iter(train))
        v0 = next(iter(val))
        _ = tok(t0["text"][:200], truncation=True, max_length=64)
        print(f"[download_c4] smoke ok train_keys={list(t0)} val_keys={list(v0)}")

    print("[download_c4] done")


if __name__ == "__main__":
    main()
