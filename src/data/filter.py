"""Filter raw Arena battles before label construction.

Per docs/train-mmbert-arena-router.md §2.4:
  - Drop rows flagged unsafe (toxic_chat_tag != 'safe' or any openai_moderation flag)
  - Dedupe by question_id (each unique conversation appears once)
  - Hash-based dedupe on lowercased+stripped first user turn (catch near-duplicates)
  - Optional: drop non-English (mmBERT is multilingual; default off)

Usage:
    uv run python -m src.data.filter \\
        --input data/raw/arena.parquet \\
        --output data/processed/arena_filtered.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def filter_battles(
    df: pd.DataFrame,
    drop_unsafe: bool = True,
    drop_non_english: bool = False,
    dedupe_near_duplicates: bool = True,
) -> pd.DataFrame:
    raise NotImplementedError("Implement on 5/8 (M2). See docs §2.4.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/raw/arena.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/arena_filtered.parquet"))
    parser.add_argument("--drop-non-english", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}")
    out = filter_battles(df, drop_non_english=args.drop_non_english)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output)
    print(f"Filtered → {len(out):,} rows → {args.output}")


if __name__ == "__main__":
    main()
