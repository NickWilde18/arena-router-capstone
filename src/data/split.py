"""Stratified 80/10/10 split → train/val/test.jsonl (docs §3).

Stratification key: argmax of soft_label.

Usage:
    uv run python -m src.data.split \\
        --input data/processed/arena_labeled.jsonl \\
        --output-dir data/processed \\
        --train 0.8 --val 0.1 --test 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split


def split_jsonl(
    input_jsonl: Path,
    output_dir: Path,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
    stratify: bool = True,
) -> dict[str, Path]:
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {train + val + test}")

    rows: list[dict] = []
    with open(input_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows):,} samples from {input_jsonl}")

    labels = [r["label"] for r in rows]

    train_rows, valtest_rows, _, valtest_labels = train_test_split(
        rows,
        labels,
        test_size=val + test,
        random_state=seed,
        stratify=labels if stratify else None,
    )
    val_ratio_in_valtest = val / (val + test)
    val_rows, test_rows = train_test_split(
        valtest_rows,
        test_size=1 - val_ratio_in_valtest,
        random_state=seed,
        stratify=valtest_labels if stratify else None,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, split_rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in split_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(split_rows):,} → {path}")
        paths[name] = path

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/processed/arena_labeled.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-stratify", action="store_true")
    args = parser.parse_args()

    split_jsonl(
        args.input,
        args.output_dir,
        args.train,
        args.val,
        args.test,
        args.seed,
        stratify=not args.no_stratify,
    )


if __name__ == "__main__":
    main()
