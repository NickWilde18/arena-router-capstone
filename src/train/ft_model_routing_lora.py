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
from transformers.trainer_pt_utils import LengthGroupedSampler

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
        out = tokenizer(batch["text"], truncation=True, max_length=max_length)
        out["length"] = [len(ids) for ids in out["input_ids"]]
        return out
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
        for f in features:
            f.pop("length", None)   # group_by_length helper, not a model input
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        batch["soft_target"] = soft
        return batch


# -----------------------------------------------------------------------------
# Trainer with switchable loss (docs §4 'Loss')
# -----------------------------------------------------------------------------


class SoftLabelTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_type: str = "kl_div",
        group_by_length: bool = False,
        compile_dynamic: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "default",
        class_weights: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if loss_type not in ("kl_div", "cross_entropy"):
            raise ValueError(f"loss_type must be kl_div|cross_entropy, got {loss_type}")
        self.loss_type = loss_type
        self._group_by_length = group_by_length
        self.class_weights = class_weights  # (num_classes,) or None
        # transformers built-in `torch_compile` recompiles per shape, which kills
        # the speedup when paired with group_by_length (many distinct seq lens).
        # Manual wrap with dynamic=True compiles a single graph that handles
        # variable shapes via symbolic dims.
        if compile_dynamic:
            self.model = torch.compile(
                self.model,
                dynamic=True,
                backend=compile_backend,
                mode=compile_mode,
            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        soft_target = inputs.pop("soft_target", None)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device, dtype=logits.dtype)
            sample_w = cw[labels]
        else:
            cw = None
            sample_w = None
        if self.loss_type == "kl_div" and soft_target is not None:
            loss = soft_cross_entropy(logits, soft_target, sample_weights=sample_w)
        else:
            loss = F.cross_entropy(logits, labels, weight=cw)
        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self, train_dataset=None):
        # transformers 5.x removed the `group_by_length` TrainingArgument; we
        # restore it via a custom sampler. Cuts padding waste 5-10x given the
        # heavy length skew (median 16 tokens, p99 532).
        if not self._group_by_length:
            return super()._get_train_sampler(train_dataset)
        ds = train_dataset if train_dataset is not None else self.train_dataset
        lengths = list(ds["length"])
        return LengthGroupedSampler(
            batch_size=self.args.train_batch_size,
            dataset=ds,
            lengths=lengths,
        )


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


def _compute_inv_freq_weights(label_ids: list[int], n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights normalized to mean=1.

    Pure 1/freq pushes per-step gradients of the rarest class up by ~10x in our
    20-class Arena split (1.1% vs 11.9% min/max). Mean-1 normalization keeps
    overall loss scale unchanged so downstream lr tuning still applies.
    """
    counts = np.bincount(label_ids, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights * (n_classes / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


def _resolve_precision(t_cfg: dict) -> dict:
    """Honor explicit bf16/fp16 from config; fall back to safe per-device default.

    Config may set bf16: true (preferred for ModernBERT) or fp16: true.
    bf16 takes precedence if both are truthy.
    """
    use_bf16 = bool(t_cfg.get("bf16", False))
    use_fp16 = bool(t_cfg.get("fp16", False))
    if torch.cuda.is_available():
        if use_bf16:
            return {"bf16": True}
        if use_fp16:
            return {"fp16": True}
        return {}
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
    # Free speedup on Ada/Hopper for fp32 matmul ops outside the bf16 autocast.
    torch.set_float32_matmul_precision("high")

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

    # --- LoRA (docs §4) or linear probe ---
    linear_probe = bool(cfg.get("lora", {}).get("linear_probe", False))
    if linear_probe:
        # Freeze the entire encoder; train only the classifier head. Used as a
        # diagnostic for whether the encoder representation is the bottleneck.
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
        if bool(cfg.get("training", {}).get("gradient_checkpointing", False)):
            model.enable_input_require_grads()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_tot = sum(p.numel() for p in model.parameters())
        print(f"[linear probe] trainable params: {n_train:,} / {n_tot:,} = {n_train/n_tot*100:.4f}%")
    else:
        l_cfg = cfg["lora"]
        peft_cfg = LoraConfig(
            r=int(l_cfg["r"]),
            lora_alpha=int(l_cfg["alpha"]),
            lora_dropout=float(l_cfg["dropout"]),
            target_modules=list(l_cfg["target_modules"]),
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, peft_cfg)
        # PEFT freezes base model; for gradient_checkpointing to propagate grads
        # through frozen layers, embeddings must opt-in to requires_grad.
        if bool(cfg.get("training", {}).get("gradient_checkpointing", False)):
            model.enable_input_require_grads()
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
    precision = _resolve_precision(t_cfg)

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
        remove_unused_columns=False,    # keep label_id + soft_target for SoftLabelCollator
        report_to="none",
        dataloader_num_workers=int(t_cfg.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(t_cfg.get("dataloader_pin_memory", True)),
        gradient_checkpointing=bool(t_cfg.get("gradient_checkpointing", False)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # transformers built-in torch_compile recompiles per shape and lacks a
        # `dynamic=True` knob; we wrap manually in SoftLabelTrainer instead.
        **precision,
    )
    compile_dynamic = bool(t_cfg.get("torch_compile", False))
    group_by_length = bool(t_cfg.get("group_by_length", False))

    loss_type = cfg.get("loss", {}).get("type", "kl_div")
    class_weight_mode = cfg.get("loss", {}).get("class_weight")
    if class_weight_mode == "inv_freq":
        class_weights = _compute_inv_freq_weights(
            list(train_ds["label_id"]), n_classes
        )
        print(f"Class weights: inv_freq (min={class_weights.min().item():.3f}, "
              f"max={class_weights.max().item():.3f}, mean={class_weights.mean().item():.3f})")
    elif class_weight_mode in (None, "null", False):
        class_weights = None
    else:
        raise ValueError(f"unknown class_weight: {class_weight_mode!r}")

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
        group_by_length=group_by_length,
        compile_dynamic=compile_dynamic,
        compile_backend=t_cfg.get("torch_compile_backend", "inductor"),
        compile_mode=t_cfg.get("torch_compile_mode", "default"),
        class_weights=class_weights,
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
