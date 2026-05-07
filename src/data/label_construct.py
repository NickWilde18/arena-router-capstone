"""Build soft labels from pairwise Arena battles (docs §2.1-2.3).

Pipeline:
  §2.1 Candidate models: keep models with frequency ≥ min_class_frequency.
  §2.2 Per-battle expansion to (prompt, label, weight) tuples:
         winner == 'model_a'        →  (prompt, m_a, 1.0)
         winner == 'model_b'        →  (prompt, m_b, 1.0)
         winner == 'tie'            →  (prompt, m_a, 0.5) + (prompt, m_b, 0.5)
         winner == 'tie (bothbad)'  →  drop
       prompt = conversation_a[0]['content'].
  §2.3 Aggregate per-prompt: P(m | prompt) = Σw·1[label==m] / Σw.

Outputs:
  data/processed/arena_labeled.jsonl   one row per unique prompt {text, label, soft_label}
  data/processed/label2id.json         deterministic class order

Usage:
    uv run python -m src.data.label_construct \\
        --input data/processed/arena_filtered.parquet \\
        --output-dir data/processed \\
        --min-class-frequency 500
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def _first_user_content(conversation) -> str:
    if conversation is None or len(conversation) == 0:
        return ""
    first = conversation[0]
    try:
        return str(first["content"])
    except (KeyError, TypeError, IndexError):
        return ""


def build_soft_labels(
    input_parquet: Path,
    output_dir: Path,
    min_class_frequency: int = 500,
    tie_weight: float = 0.5,
) -> tuple[Path, Path]:
    df = pd.read_parquet(input_parquet)
    print(f"Loaded {len(df):,} battles from {input_parquet}\n")

    # §2.1 candidate models
    all_models = pd.concat([df["model_a"], df["model_b"]], ignore_index=True)
    counts = all_models.value_counts()
    kept_models = sorted(counts[counts >= min_class_frequency].index.tolist())
    dropped_models = counts[counts < min_class_frequency].index.tolist()
    label2id = {m: i for i, m in enumerate(kept_models)}
    kept_set = set(kept_models)
    print(f"§2.1 Models with ≥ {min_class_frequency} battles: {len(kept_models)} kept")
    print(f"     Kept   : {kept_models}")
    print(f"     Dropped: {dropped_models}\n")

    # §2.2 expand
    samples: list[tuple[str, str, float]] = []
    n_bothbad = n_oov = n_unknown = n_empty = 0
    for row in df.itertuples(index=False):
        prompt = _first_user_content(row.conversation_a)
        if not prompt:
            n_empty += 1
            continue
        winner = row.winner
        m_a, m_b = row.model_a, row.model_b
        if winner == "tie (bothbad)":
            n_bothbad += 1
        elif winner == "model_a":
            if m_a in kept_set:
                samples.append((prompt, m_a, 1.0))
            else:
                n_oov += 1
        elif winner == "model_b":
            if m_b in kept_set:
                samples.append((prompt, m_b, 1.0))
            else:
                n_oov += 1
        elif winner == "tie":
            kept_pair = []
            if m_a in kept_set:
                kept_pair.append((prompt, m_a, tie_weight))
            if m_b in kept_set:
                kept_pair.append((prompt, m_b, tie_weight))
            if kept_pair:
                samples.extend(kept_pair)
            else:
                n_oov += 1
        else:
            n_unknown += 1

    print(f"§2.2 Expanded into {len(samples):,} (prompt, label, weight) tuples")
    print(f"     dropped tie(bothbad)   : {n_bothbad:,}")
    print(f"     dropped OOV-model      : {n_oov:,}")
    print(f"     dropped empty prompt   : {n_empty:,}")
    if n_unknown:
        print(f"     dropped unknown winner : {n_unknown:,}")
    print()

    # §2.3 aggregate to soft labels
    agg: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for prompt, label, weight in samples:
        agg[prompt][label] += weight

    rows: list[dict] = []
    for prompt, label_weights in agg.items():
        total = sum(label_weights.values())
        soft_label = {m: w / total for m, w in label_weights.items()}
        argmax_label = max(soft_label.items(), key=lambda kv: kv[1])[0]
        rows.append({"text": prompt, "label": argmax_label, "soft_label": soft_label})

    print(f"§2.3 Aggregated to {len(rows):,} unique-prompt samples")
    label_dist = pd.Series([r["label"] for r in rows]).value_counts()
    print("     argmax-label distribution:")
    for m, n in label_dist.items():
        print(f"       {m:<32s}  {n:>6,}  ({n/len(rows)*100:5.1f}%)")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)
    label2id_path = output_dir / "label2id.json"
    with open(label2id_path, "w") as f:
        json.dump(label2id, f, indent=2)

    labeled_path = output_dir / "arena_labeled.jsonl"
    with open(labeled_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {labeled_path}")
    print(f"Wrote {label2id_path}")
    return labeled_path, label2id_path


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
