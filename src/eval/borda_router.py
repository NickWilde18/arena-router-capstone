"""Aggregate the pairwise classifier into a top-K routing decision (L3 #7).

Given a pairwise classifier `f(prompt, m_a, m_b) → P(a wins)`, route a prompt
to the top-K-backend pool by:

  1. For each ordered pair (i, j), i ≠ j, query the classifier with input
     `[a={m_i}] [b={m_j}] {prompt}` and extract P(i wins).
  2. Score(m) = Σ_{j ≠ m} P(m wins | m, j)   (Borda-count style).
  3. argmax over models = routed pick.

Compare against the **5-class router's gold label** on its own test set so
the numbers are directly comparable to `reports/top5_class_restriction.md`
(test top1 = 0.2380, exp_winrate = 0.2384).

Usage:
    uv run python -m src.eval.borda_router \\
        --adapter outputs/pairwise_top5/final \\
        --eval-split test \\
        --eval-data-dir data/processed/top5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _load(adapter_dir: Path, base: str):
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tok


@torch.no_grad()
def _pairwise_probs(
    model,
    tok,
    prompts: list[str],
    pairs: list[tuple[str, str]],
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 512,
) -> np.ndarray:
    """Returns (N_prompts, N_pairs) array of P(label='a' wins) for each (prompt, (a, b)) cell."""
    n_p = len(prompts)
    n_pair = len(pairs)
    out = np.zeros((n_p, n_pair), dtype=np.float32)
    flat_inputs: list[str] = []
    flat_idx: list[tuple[int, int]] = []
    for i, prompt in enumerate(prompts):
        for j, (m_a, m_b) in enumerate(pairs):
            flat_inputs.append(f"[a={m_a}] [b={m_b}] {prompt}")
            flat_idx.append((i, j))

    for start in range(0, len(flat_inputs), batch_size):
        batch = flat_inputs[start:start + batch_size]
        enc = tok(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        # label2id for pairwise: {"a": 0, "b": 1} (see build_pairwise.py)
        pa = F.softmax(logits.float(), dim=-1)[:, 0].cpu().numpy()
        for k, prob_a in enumerate(pa):
            i, j = flat_idx[start + k]
            out[i, j] = prob_a
    return out


def borda_aggregate(prob_a: np.ndarray, pairs: list[tuple[str, str]], models: list[str]) -> np.ndarray:
    """Per-model Borda score from the pairwise probability matrix."""
    n_p = prob_a.shape[0]
    model_to_idx = {m: i for i, m in enumerate(models)}
    scores = np.zeros((n_p, len(models)), dtype=np.float32)
    for j, (m_a, m_b) in enumerate(pairs):
        ia, ib = model_to_idx[m_a], model_to_idx[m_b]
        scores[:, ia] += prob_a[:, j]
        scores[:, ib] += 1.0 - prob_a[:, j]
    return scores


def evaluate(
    adapter_dir: Path,
    eval_jsonl: Path,
    label2id_path: Path,
    base: str = "jhu-clsp/mmBERT-base",
    batch_size: int = 64,
    max_length: int = 512,
) -> dict:
    with open(label2id_path) as f:
        label2id: dict[str, int] = json.load(f)
    id2label = {i: m for m, i in label2id.items()}
    models = list(label2id.keys())

    rows = [json.loads(line) for line in open(eval_jsonl) if line.strip()]
    prompts = [r["text"] for r in rows]
    soft_targets = np.zeros((len(rows), len(models)), dtype=np.float32)
    labels = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        labels[i] = label2id[r["label"]]
        for m, p in r["soft_label"].items():
            if m in label2id:
                soft_targets[i, label2id[m]] = float(p)

    pairs = [(models[a], models[b]) for a in range(len(models)) for b in range(len(models)) if a != b]
    print(f"Evaluating {len(rows):,} prompts × {len(pairs)} ordered pairs ({len(models)}-class)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = _load(adapter_dir, base)
    model.to(device)
    prob_a = _pairwise_probs(model, tok, prompts, pairs, device, batch_size, max_length)
    scores = borda_aggregate(prob_a, pairs, models)
    preds = scores.argmax(axis=-1)

    top1 = float((preds == labels).mean())
    idx = np.arange(len(preds))
    expected_winrate = float(soft_targets[idx, preds].mean())
    pred_dist = Counter(int(p) for p in preds)
    label_dist = Counter(int(l) for l in labels)
    majority_class = max(label_dist, key=label_dist.get)
    majority_top1 = label_dist[majority_class] / len(labels)

    return {
        "n_eval": len(rows),
        "n_models": len(models),
        "n_pairs_ordered": len(pairs),
        "top1_accuracy": top1,
        "expected_winrate": expected_winrate,
        "majority_class": id2label[majority_class],
        "majority_top1": majority_top1,
        "delta_top1_vs_majority": top1 - majority_top1,
        "pred_distribution": {id2label[c]: pred_dist.get(c, 0) for c in range(len(models))},
        "per_class_recall": {
            id2label[c]: (
                int(((preds == c) & (labels == c)).sum()) / max(label_dist.get(c, 0), 1)
            )
            for c in range(len(models))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", type=Path, required=True, help="pairwise adapter dir (final/)")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--eval-data-dir", type=Path, default=Path("data/processed/top5"))
    parser.add_argument("--base-model", type=str, default="jhu-clsp/mmBERT-base")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--json", type=Path, help="optional JSON output path")
    args = parser.parse_args()

    eval_jsonl = args.eval_data_dir / f"{args.eval_split}.jsonl"
    label2id_path = args.eval_data_dir / "label2id.json"
    metrics = evaluate(
        args.adapter, eval_jsonl, label2id_path,
        base=args.base_model, batch_size=args.batch_size, max_length=args.max_length,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
