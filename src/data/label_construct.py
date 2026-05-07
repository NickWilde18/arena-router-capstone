"""Build soft labels from pairwise Arena battles (docs §2.2-2.3).

Pipeline:
  1. For each battle, emit (prompt, label, weight) tuples per §2.2:
       winner == 'model_a'         →  (prompt, m_a, 1.0)
       winner == 'model_b'         →  (prompt, m_b, 1.0)
       winner == 'tie'             →  (prompt, m_a, 0.5) + (prompt, m_b, 0.5)
       winner == 'tie (bothbad)'   →  drop
  2. Group by prompt → soft label  P(m | prompt) ∈ Δ^N
  3. Filter classes by min-frequency threshold (default 500); keep top-N models
  4. Persist label2id.json + arena_labeled.jsonl  (text, label, soft_label)

Usage:
    uv run python -m src.data.label_construct \\
        --input data/processed/arena_filtered.parquet \\
        --output-dir data/processed \\
        --min-class-frequency 500
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_soft_labels(
    input_parquet: Path,
    output_dir: Path,
    min_class_frequency: int = 500,
    tie_weight: float = 0.5,
) -> tuple[Path, Path]:
    raise NotImplementedError("Implement on 5/8 (M3). See docs §2.2-2.3.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/processed/arena_filtered.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-class-frequency", type=int, default=500)
    parser.add_argument("--tie-weight", type=float, default=0.5)
    args = parser.parse_args()

    build_soft_labels(args.input, args.output_dir, args.min_class_frequency, args.tie_weight)


if __name__ == "__main__":
    main()
