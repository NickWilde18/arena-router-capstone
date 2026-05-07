"""LoRA fine-tune mmBERT for prompt → best-model classification (docs §4).

Implementation notes (departures from docs §4 due to engineering reality):
  - transformers 5.x renamed `evaluation_strategy` → `eval_strategy`
  - `fp16` is honored on CUDA but auto-replaced with `bf16` on MPS (fp16 unstable
    on Apple silicon); CPU runs full precision.
  - Custom Trainer with switchable loss: 'kl_div' (soft) | 'cross_entropy' (hard)
  - Custom collator pads (text, label_id, soft_target) jointly.
  - compute_metrics returns top-1 accuracy + expected_winrate (docs §4).
  - --smoke: 1k train rows / 100 max_steps, for MPS sanity (~5 min on M-series).

Usage:
    uv run python -m src.train.ft_model_routing_lora --config configs/train.yaml
    uv run python -m src.train.ft_model_routing_lora --config configs/train.yaml --smoke
    uv run python -m src.train.ft_model_routing_lora --loss cross_entropy   # ablation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from .soft_loss import soft_cross_entropy


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def _build_dataset(jsonl_path: Path, label2id: dict[str, int]) -> Dataset:
    """Load JSONL → Dataset with text / label_id / soft_target (dense vector)."""
    n_classes = len(label2id)
    records: list[dict[str, Any]] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            soft = [0.0] * n_classes
            for m, p in ex["soft_label"].items():
                if m in label2id:
                    soft[label2id[m]] = float(p)
            records.append({
                "text": ex["text"],
                "label_id": int(label2id[ex["label"]]),
                "soft_target": soft,
            })
    return Dataset.from_list(records)


def _tokenize(ds: Dataset, tokenizer: PreTrainedTokenizerBase, max_length: int) -> Dataset:
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)
    return ds.map(_tok, batched=True, remove_columns=["text"])


# -----------------------------------------------------------------------------
# Collator: pad text + carry soft_target / label_id
# -----------------------------------------------------------------------------


@dataclass
class SoftLabelCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        soft = torch.tensor([f.pop("soft_target") for f in features], dtype=torch.float32)
        labels = torch.tensor([f.pop("label_id") for f in features], dtype=torch.long)
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        batch["soft_target"] = soft
        return batch


# -----------------------------------------------------------------------------
# Trainer with switchable loss (docs §4 'Loss')
# -----------------------------------------------------------------------------


class SoftLabelTrainer(Trainer):
    def __init__(self, *args, loss_type: str = "kl_div", **kwargs):
        super().__init__(*args, **kwargs)
        if loss_type not in ("kl_div", "cross_entropy"):
            raise ValueError(f"loss_type must be kl_div|cross_entropy, got {loss_type}")
        self.loss_type = loss_type

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        soft_target = inputs.pop("soft_target", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.loss_type == "kl_div" and soft_target is not None:
            loss = soft_cross_entropy(logits, soft_target)
        else:
            loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


# -----------------------------------------------------------------------------
# Metrics (docs §4 'Evaluation metrics')
# -----------------------------------------------------------------------------


def make_compute_metrics(eval_dataset: Dataset):
    """top1_accuracy + expected_winrate.

    Trainer feeds the eval dataset in order without shuffle, so we look up
    soft_target by row index aligned with predictions.
    """
    soft_targets = np.array(eval_dataset["soft_target"], dtype=np.float32)

    def _fn(eval_pred):
        logits, label_ids = eval_pred.predictions, eval_pred.label_ids
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = logits.argmax(axis=-1)
        top1 = float((preds == label_ids).mean())
        idx = np.arange(len(preds))
        expected_winrate = float(soft_targets[idx, preds].mean())
        return {"top1_accuracy": top1, "expected_winrate": expected_winrate}

    return _fn


# -----------------------------------------------------------------------------
# Precision: pick what works on the active accelerator
# -----------------------------------------------------------------------------


def _resolve_precision(use_fp16: bool) -> dict:
    if torch.cuda.is_available():
        return {"fp16": use_fp16}
    if torch.backends.mps.is_available():
        return {"bf16": True}     # fp16 unstable on MPS
    return {}


# -----------------------------------------------------------------------------
# train()
# -----------------------------------------------------------------------------


def train(cfg: dict, smoke: bool = False) -> Path:
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- labels ---
    with open(cfg["data"]["label2id_file"]) as f:
        label2id: dict[str, int] = json.load(f)
    id2label = {i: m for m, i in label2id.items()}
    n_classes = len(label2id)
    print(f"Classes: {n_classes}")

    # --- model + tokenizer ---
    base = cfg["model"]["base"]
    max_length = int(cfg["model"]["max_length"])
    print(f"Base model: {base}  (max_length={max_length})")
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base,
        num_labels=n_classes,
        id2label=id2label,
        label2id=label2id,
    )

    # --- LoRA (docs §4) ---
    l_cfg = cfg["lora"]
    peft_cfg = LoraConfig(
        r=int(l_cfg["r"]),
        lora_alpha=int(l_cfg["alpha"]),
        lora_dropout=float(l_cfg["dropout"]),
        target_modules=list(l_cfg["target_modules"]),
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    # --- datasets ---
    train_ds = _build_dataset(Path(cfg["data"]["train_file"]), label2id)
    val_ds = _build_dataset(Path(cfg["data"]["val_file"]), label2id)
    if smoke:
        train_ds = train_ds.select(range(min(1000, len(train_ds))))
        val_ds = val_ds.select(range(min(200, len(val_ds))))
        print(f"[smoke] train={len(train_ds)}, val={len(val_ds)}")
    train_ds = _tokenize(train_ds, tokenizer, max_length)
    val_ds = _tokenize(val_ds, tokenizer, max_length)

    # --- training args ---
    t_cfg = cfg["training"]
    output_dir = Path(t_cfg["output_dir"])
    if smoke:
        output_dir = output_dir.parent / (output_dir.name + "_smoke")
    precision = _resolve_precision(bool(t_cfg.get("fp16", True)))

    eval_strategy = t_cfg.get("eval_strategy", t_cfg.get("evaluation_strategy", "steps"))
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1 if smoke else int(t_cfg["num_train_epochs"]),
        max_steps=100 if smoke else -1,
        per_device_train_batch_size=int(t_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(t_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(t_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(t_cfg["learning_rate"]),
        warmup_ratio=float(t_cfg["warmup_ratio"]),
        weight_decay=float(t_cfg["weight_decay"]),
        eval_strategy=eval_strategy,
        eval_steps=50 if smoke else int(t_cfg.get("eval_steps", 500)),
        save_strategy="steps",
        save_steps=100 if smoke else int(t_cfg.get("save_steps", 500)),
        logging_steps=10 if smoke else int(t_cfg.get("logging_steps", 50)),
        save_total_limit=2,
        load_best_model_at_end=(not smoke) and bool(t_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=t_cfg.get("metric_for_best_model", "top1_accuracy"),
        greater_is_better=bool(t_cfg.get("greater_is_better", True)),
        seed=seed,
        report_to="none",
        **precision,
    )

    loss_type = cfg.get("loss", {}).get("type", "kl_div")
    print(f"Loss type:  {loss_type}")
    print(f"Precision:  {precision or 'fp32'}")
    print(f"Output dir: {output_dir}")

    trainer = SoftLabelTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=SoftLabelCollator(tokenizer=tokenizer),
        compute_metrics=make_compute_metrics(val_ds),
        loss_type=loss_type,
    )

    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved adapter → {final_dir}")
    return final_dir


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--smoke", action="store_true", help="1k samples / 100 steps for MPS sanity")
    parser.add_argument("--loss", choices=["kl_div", "cross_entropy"], help="Override config loss type")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.loss:
        cfg.setdefault("loss", {})["type"] = args.loss
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
