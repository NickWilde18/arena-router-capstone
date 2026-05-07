"""Restrict the labeled dataset to the top-K most frequent classes.

Why: post-hoc analysis of saved adapters shows the model never predicts 14
of the 20 classes (mode collapse to {vicuna-13b, gpt-4} basin). The 14
unpredicted classes also contribute most of the soft-label noise via tie
splits. For VSR routing, only backends actually deployed need to be
predicted - a top-K restriction matches the production assumption and
gives a denser, less-noisy training signal.

For each row:
  1. drop soft_label entries not in the top-K class set
  2. drop rows whose remaining kept-mass is below `--min-kept-mass`
     (default 0.5 - at least half the original soft mass survived)
  3. renormalize remaining soft_label entries to sum to 1
  4. recompute argmax label

Outputs a new `arena_labeled_topK.jsonl` + `label2id_topK.json`. Run the
existing split script on top to produce {train,val,test}.jsonl.

Usage:
    uv run python -m src.data.restrict_top_k \\
        --input data/processed/arena_labeled.jsonl \\
        --output-dir data/processed/top5 \\
        --top-k 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def restrict(
    input_jsonl: Path,
    output_dir: Path,
    top_k: int,
    min_kept_mass: float = 0.5,
) -> tuple[Path, Path]:
    rows = [json.loads(line) for line in open(input_jsonl) if line.strip()]
    print(f"Loaded {len(rows):,} rows from {input_jsonl}")

    label_counts: Counter = Counter()
    for r in rows:
        label_counts[r["label"]] += 1
    kept_classes = sorted(c for c, _ in label_counts.most_common(top_k))
    print(f"Top-{top_k} classes by argmax frequency:")
    for c in sorted(kept_classes, key=lambda c: -label_counts[c]):
        print(f"  {c:30s} {label_counts[c]:>6,}  ({label_counts[c]/len(rows)*100:5.1f}%)")
    kept_set = set(kept_classes)

    new_rows: list[dict] = []
    n_drop_low_mass = 0
    for r in rows:
        kept = {m: p for m, p in r["soft_label"].items() if m in kept_set}
        kept_mass = sum(kept.values())
        if kept_mass < min_kept_mass:
            n_drop_low_mass += 1
            continue
        renorm = {m: p / kept_mass for m, p in kept.items()}
        argmax_label = max(renorm.items(), key=lambda kv: kv[1])[0]
        new_rows.append({"text": r["text"], "label": argmax_label, "soft_label": renorm})

    print(f"Kept {len(new_rows):,} / {len(rows):,} rows "
          f"(dropped {n_drop_low_mass:,} with kept-mass < {min_kept_mass})")

    new_label_counts: Counter = Counter(r["label"] for r in new_rows)
    print(f"New argmax-label distribution:")
    for c in sorted(kept_classes, key=lambda c: -new_label_counts[c]):
        print(f"  {c:30s} {new_label_counts[c]:>6,}  "
              f"({new_label_counts[c]/len(new_rows)*100:5.1f}%)")

    output_dir.mkdir(parents=True, exist_ok=True)
    label2id = {m: i for i, m in enumerate(kept_classes)}
    label2id_path = output_dir / "label2id.json"
    with open(label2id_path, "w") as f:
        json.dump(label2id, f, indent=2)

    labeled_path = output_dir / "arena_labeled.jsonl"
    with open(labeled_path, "w") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {labeled_path}")
    print(f"Wrote {label2id_path}")
    return labeled_path, label2id_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/processed/arena_labeled.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-kept-mass", type=float, default=0.5)
    args = parser.parse_args()
    restrict(args.input, args.output_dir, args.top_k, args.min_kept_mass)


if __name__ == "__main__":
    main()
