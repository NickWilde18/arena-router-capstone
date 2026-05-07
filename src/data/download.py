"""Download lmsys/chatbot_arena_conversations and persist as parquet.

The dataset is gated. Request access at
https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
then export HF_TOKEN=hf_xxx before running.

Usage:
    uv run python -m src.data.download
    uv run python -m src.data.download --output-dir data/raw
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "lmsys/chatbot_arena_conversations"


def download(output_dir: Path, revision: str = "main") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN env var not set. The Arena dataset is gated; "
            "request access at https://huggingface.co/datasets/lmsys/chatbot_arena_conversations "
            "then export HF_TOKEN=hf_xxx before running."
        )

    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, revision=revision, token=token, split="train")
    out_file = output_dir / "arena.parquet"
    ds.to_parquet(out_file)
    print(f"Wrote {len(ds):,} rows → {out_file}")
    print(f"Columns: {ds.column_names}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    download(args.output_dir, args.revision)


if __name__ == "__main__":
    main()
