"""Filter raw Arena battles before label construction.

Per docs/train-mmbert-arena-router.md §2.4:
  - Dedupe by question_id
  - Drop rows where toxic_chat_tag != 'safe'
  - Optional: drop non-English (mmBERT is multilingual; default off)
  - Hash-based dedupe on lowercased+stripped first user turn

Usage:
    uv run python -m src.data.filter \\
        --input data/raw/arena.parquet \\
        --output data/processed/arena_filtered.parquet
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def _first_user_content(conversation) -> str:
    """Extract conversation_a[0]['content'] (first user turn) per docs §2.2."""
    if conversation is None or len(conversation) == 0:
        return ""
    first = conversation[0]
    try:
        return str(first["content"])
    except (KeyError, TypeError, IndexError):
        return ""


def filter_battles(
    df: pd.DataFrame,
    drop_unsafe: bool = True,
    drop_non_english: bool = False,
    dedupe_near_duplicates: bool = True,
) -> pd.DataFrame:
    n0 = len(df)
    print(f"Input: {n0:,} rows")

    if "language" in df.columns:
        print(f"  language top-5: {df['language'].value_counts().head(5).to_dict()}")
    if "toxic_chat_tag" in df.columns:
        print(f"  toxic_chat_tag: {df['toxic_chat_tag'].value_counts().to_dict()}")
    if "winner" in df.columns:
        print(f"  winner        : {df['winner'].value_counts().to_dict()}")

    if "question_id" in df.columns:
        df = df.drop_duplicates(subset=["question_id"]).reset_index(drop=True)
        print(f"  After question_id dedup           : {len(df):,}")

    if drop_unsafe and "toxic_chat_tag" in df.columns:
        df = df[df["toxic_chat_tag"] == "safe"].reset_index(drop=True)
        print(f"  After toxic_chat_tag == 'safe'    : {len(df):,}")

    if drop_non_english and "language" in df.columns:
        df = df[df["language"].astype(str).str.lower().isin({"english", "en"})].reset_index(drop=True)
        print(f"  After language == English/en      : {len(df):,}")

    if dedupe_near_duplicates:
        prompts = df["conversation_a"].apply(_first_user_content)
        normalized = prompts.str.lower().str.strip()
        hashes = normalized.apply(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())
        df = df[~hashes.duplicated()].reset_index(drop=True)
        print(f"  After near-dup hash on first turn : {len(df):,}")

    print(f"Output: {len(df):,} rows ({len(df)/max(n0,1)*100:.1f}% kept)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/raw/arena.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/arena_filtered.parquet"))
    parser.add_argument("--drop-non-english", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} rows from {args.input}\n")
    out = filter_battles(df, drop_non_english=args.drop_non_english)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output)
    print(f"\nWrote → {args.output}")


if __name__ == "__main__":
    main()
