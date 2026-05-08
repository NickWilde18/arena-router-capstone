"""Build per-battle pairwise dataset (L3 #7 reframing).

Each Arena battle becomes a single (prompt, m_a, m_b) -> binary {a, b} sample.
Removes the soft-label aggregation that diluted multi-class training and
produces ~1.4× more training rows than the aggregated 22,410-prompt set.

Input format (concatenated into one text field for compatibility with the
existing classifier pipeline):

    [a=<model_a>] [b=<model_b>] <prompt>

Filters:
  - winner ∈ {model_a, model_b} (drop ties and tie(bothbad))
  - both model_a and model_b in the top-K kept set
  - prompt non-empty
  - applies the same toxic_chat_tag / question_id dedup as the aggregated path

Outputs:
  data/processed/pairwise/arena_labeled.jsonl  one row per battle
  data/processed/pairwise/label2id.json        {"a": 0, "b": 1}

Usage:
    uv run python -m src.data.build_pairwise \\
        --input data/processed/arena_filtered.parquet \\
        --output-dir data/processed/pairwise \\
        --top-k 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def build_pairwise(
    input_parquet: Path,
    output_dir: Path,
    top_k: int = 5,
    min_class_frequency: int = 500,
) -> tuple[Path, Path]:
    df = pd.read_parquet(input_parquet)
    print(f"Loaded {len(df):,} battles from {input_parquet}")

    # By default match the top-K-by-aggregated-argmax-frequency set used in
    # the 5-class router (data/processed/top5/label2id.json). Falls back to
    # raw battle frequency only if the explicit list is unavailable.
    explicit_top_k = Path("data/processed/top5/label2id.json")
    if top_k == 5 and explicit_top_k.exists():
        with open(explicit_top_k) as f:
            kept_models = list(json.load(f).keys())
        print(f"Using explicit top-5 from {explicit_top_k}: {sorted(kept_models)}")
    else:
        all_models = pd.concat([df["model_a"], df["model_b"]], ignore_index=True)
        counts = all_models.value_counts()
        if top_k:
            kept_models = counts.head(top_k).index.tolist()
        else:
            kept_models = counts[counts >= min_class_frequency].index.tolist()
        print(f"Kept models ({len(kept_models)}) by raw battle count: {sorted(kept_models)}")
    kept_set = set(kept_models)

    rows: list[dict] = []
    n_drop_unknown_winner = n_drop_tie = n_drop_bothbad = n_drop_oov = n_drop_empty = 0
    for r in df.itertuples(index=False):
        prompt = _first_user_content(r.conversation_a)
        if not prompt:
            n_drop_empty += 1
            continue
        m_a, m_b = r.model_a, r.model_b
        if m_a not in kept_set or m_b not in kept_set:
            n_drop_oov += 1
            continue
        winner = r.winner
        if winner == "model_a":
            label = "a"
        elif winner == "model_b":
            label = "b"
        elif winner == "tie":
            n_drop_tie += 1
            continue
        elif winner == "tie (bothbad)":
            n_drop_bothbad += 1
            continue
        else:
            n_drop_unknown_winner += 1
            continue

        text = f"[a={m_a}] [b={m_b}] {prompt}"
        rows.append({"text": text, "label": label, "model_a": m_a, "model_b": m_b})

    print(f"\nKept {len(rows):,} pairwise samples")
    print(f"  dropped empty:           {n_drop_empty:,}")
    print(f"  dropped OOV-pair:        {n_drop_oov:,}")
    print(f"  dropped tie:             {n_drop_tie:,}")
    print(f"  dropped tie(bothbad):    {n_drop_bothbad:,}")
    print(f"  dropped unknown winner:  {n_drop_unknown_winner:,}")

    label_counts = Counter(r["label"] for r in rows)
    print(f"\nLabel distribution: {dict(label_counts)}")

    pair_counts: Counter = Counter()
    for r in rows:
        pair = tuple(sorted([r["model_a"], r["model_b"]]))
        pair_counts[pair] += 1
    print(f"\nPair distribution (top 10):")
    for pair, n in pair_counts.most_common(10):
        print(f"  {pair[0]:25s} vs {pair[1]:25s}: {n:,}")

    output_dir.mkdir(parents=True, exist_ok=True)
    label2id = {"a": 0, "b": 1}
    label2id_path = output_dir / "label2id.json"
    with open(label2id_path, "w") as f:
        json.dump(label2id, f, indent=2)

    labeled_path = output_dir / "arena_labeled.jsonl"
    with open(labeled_path, "w") as f:
        for r in rows:
            # soft_label for compatibility with _build_dataset (1-hot on label)
            soft = {r["label"]: 1.0}
            f.write(json.dumps({
                "text": r["text"],
                "label": r["label"],
                "soft_label": soft,
                "model_a": r["model_a"],
                "model_b": r["model_b"],
            }, ensure_ascii=False) + "\n")
    print(f"\nWrote {labeled_path}")
    print(f"Wrote {label2id_path}")
    return labeled_path, label2id_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/processed/arena_filtered.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/pairwise"))
    parser.add_argument("--top-k", type=int, default=5,
                        help="keep top-K models by frequency; set 0 to use min_class_frequency instead")
    parser.add_argument("--min-class-frequency", type=int, default=500)
    args = parser.parse_args()
    build_pairwise(args.input, args.output_dir, args.top_k, args.min_class_frequency)


if __name__ == "__main__":
    main()
