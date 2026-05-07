"""Stratified 80/10/10 split → train/val/test.jsonl.

Stratification key: argmax of soft_label (per docs §3 default hard label).

Usage:
    uv run python -m src.data.split \\
        --input data/processed/arena_labeled.jsonl \\
        --output-dir data/processed \\
        --train 0.8 --val 0.1 --test 0.1 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path


def split_jsonl(
    input_jsonl: Path,
    output_dir: Path,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> dict[str, Path]:
    raise NotImplementedError("Implement on 5/8 (M3).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/processed/arena_labeled.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_jsonl(args.input, args.output_dir, args.train, args.val, args.test, args.seed)


if __name__ == "__main__":
    main()
