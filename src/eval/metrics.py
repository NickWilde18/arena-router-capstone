"""Routing-quality metrics (docs §4 'Evaluation metrics').

  - top1_accuracy:    prediction matches argmax of soft_label
  - expected_winrate: mean over eval set of soft_label[pred_class];
                      estimated probability that the routed model wins.
                      Upper-bound approximation of real routing payoff.
  - ECE:              expected calibration error (15-bin)
  - per-class report: predictions issued / hits / coverage per class
  - confusion top-K:  most common true→pred swaps, for error analysis

Usage:
    uv run python -m src.eval.metrics --adapter outputs/<run>/final
    uv run python -m src.eval.metrics --adapter <dir> --split val
    uv run python -m src.eval.metrics --adapter <dir> --split test --json reports/<run>.json
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


def _load_model(adapter_dir: Path, base_model: str, n_classes: int):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=n_classes)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model, tokenizer


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def _predict_all(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probs, preds) where probs is (N, n_classes) softmax."""
    model.to(device)
    all_probs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        all_probs.append(probs)
    P = np.concatenate(all_probs, axis=0)
    return P, P.argmax(axis=-1)


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error against hard labels (argmax of soft target)."""
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    correct = (predictions == labels).astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate(
    adapter_dir: Path,
    eval_jsonl: Path,
    label2id_path: Path,
    base_model: str = "jhu-clsp/mmBERT-base",
    batch_size: int = 32,
    max_length: int = 512,
    confusion_top_k: int = 10,
) -> dict:
    with open(label2id_path) as f:
        label2id: dict[str, int] = json.load(f)
    id2label = {i: m for m, i in label2id.items()}
    n_classes = len(label2id)

    rows = _read_jsonl(eval_jsonl)
    texts = [r["text"] for r in rows]
    labels = np.array([label2id[r["label"]] for r in rows], dtype=np.int64)
    soft_targets = np.zeros((len(rows), n_classes), dtype=np.float32)
    for i, r in enumerate(rows):
        for m, p in r["soft_label"].items():
            if m in label2id:
                soft_targets[i, label2id[m]] = float(p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_model(adapter_dir, base_model, n_classes)
    probs, preds = _predict_all(model, tokenizer, texts, device, batch_size, max_length)

    top1 = float((preds == labels).mean())
    idx = np.arange(len(preds))
    expected_winrate = float(soft_targets[idx, preds].mean())
    ece = _ece(probs, labels)

    pred_counter = Counter(int(p) for p in preds)
    label_counter = Counter(int(l) for l in labels)
    coverage = {
        id2label[c]: {
            "label_count": label_counter.get(c, 0),
            "pred_count": pred_counter.get(c, 0),
            "hits": int(((preds == c) & (labels == c)).sum()),
        }
        for c in range(n_classes)
    }
    per_class_recall = {
        m: (info["hits"] / info["label_count"]) if info["label_count"] else None
        for m, info in coverage.items()
    }

    confusion_pairs: Counter = Counter()
    for t, p in zip(labels, preds):
        if t != p:
            confusion_pairs[(int(t), int(p))] += 1
    top_confusion = [
        {"true": id2label[t], "pred": id2label[p], "count": c}
        for (t, p), c in confusion_pairs.most_common(confusion_top_k)
    ]

    majority_class = max(label_counter, key=label_counter.get)
    majority_top1 = label_counter[majority_class] / len(labels)

    return {
        "n_eval": len(rows),
        "top1_accuracy": top1,
        "expected_winrate": expected_winrate,
        "ece": ece,
        "majority_class": id2label[majority_class],
        "majority_top1": majority_top1,
        "delta_top1_vs_majority": top1 - majority_top1,
        "pred_distribution": {id2label[c]: pred_counter.get(c, 0) for c in range(n_classes)},
        "per_class_recall": per_class_recall,
        "top_confusion_pairs": top_confusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", type=Path, required=True, help="path to adapter dir (the `final/` subdir)")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--base-model", type=str, default="jhu-clsp/mmBERT-base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--json", type=Path, help="optional JSON output path")
    args = parser.parse_args()

    eval_jsonl = args.data_dir / f"{args.split}.jsonl"
    label2id_path = args.data_dir / "label2id.json"
    metrics = evaluate(
        args.adapter, eval_jsonl, label2id_path,
        base_model=args.base_model, batch_size=args.batch_size, max_length=args.max_length,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
