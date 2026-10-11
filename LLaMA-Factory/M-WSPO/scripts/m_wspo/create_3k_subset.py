#!/usr/bin/env python3
"""Create a reproducible 3K RLHF-V subset and aligned weak-logprob cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/rlhf_v_mwspo.json"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("cache/rlhf_v_mwspo_weak_lp_qwen3b.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/rlhf_v_mwspo_3k.json"))
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=Path("cache/rlhf_v_mwspo_3k_weak_lp_qwen3b.parquet"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/rlhf_v_mwspo_3k_manifest.json"),
    )
    parser.add_argument("--size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1994)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.size <= 0 or args.size >= len(rows):
        raise ValueError(f"--size must be in [1, {len(rows) - 1}], got {args.size}")

    cache = pd.read_parquet(args.cache).set_index("sample_id", drop=False)
    expected_ids = set(range(len(rows)))
    cache_ids = set(int(value) for value in cache.index)
    if cache_ids != expected_ids:
        missing = sorted(expected_ids - cache_ids)[:10]
        extra = sorted(cache_ids - expected_ids)[:10]
        raise ValueError(f"Cache IDs do not match dataset rows; missing={missing}, extra={extra}")

    rng = random.Random(args.seed)
    selected_ids = rng.sample(range(len(rows)), args.size)
    selected_set = set(selected_ids)
    heldout_ids = [idx for idx in range(len(rows)) if idx not in selected_set]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for original_id in selected_ids:
            handle.write(json.dumps(rows[original_id], ensure_ascii=False) + "\n")

    subset_cache = cache.loc[selected_ids].copy().reset_index(drop=True)
    subset_cache["original_sample_id"] = selected_ids
    subset_cache["sample_id"] = range(args.size)
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    subset_cache.to_parquet(args.output_cache, index=False)

    missing_images: list[str] = []
    repo_root = Path.cwd()
    for original_id in selected_ids:
        for image in rows[original_id].get("images", []):
            if not (repo_root / image).is_file():
                missing_images.append(image)
    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} subset images are missing; examples: {missing_images[:5]}")

    manifest = {
        "source_dataset": str(args.input),
        "source_dataset_sha256": sha256(args.input),
        "source_cache": str(args.cache),
        "source_cache_sha256": sha256(args.cache),
        "source_size": len(rows),
        "subset_size": args.size,
        "subset_fraction": args.size / len(rows),
        "seed": args.seed,
        "selected_original_sample_ids": selected_ids,
        "heldout_original_sample_ids": heldout_ids,
        "subset_dataset": str(args.output),
        "subset_dataset_sha256": sha256(args.output),
        "subset_cache": str(args.output_cache),
        "subset_cache_sha256": sha256(args.output_cache),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.size} pairs ({args.size / len(rows):.2%}) to {args.output}")
    print(f"Wrote aligned cache with IDs 0..{args.size - 1} to {args.output_cache}")
    print(f"Wrote {len(heldout_ids)} held-out IDs to {args.manifest}")


if __name__ == "__main__":
    main()
