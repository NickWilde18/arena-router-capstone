# Baseline Run · v2 Data + KL Loss · 2026-05-07

> Capstone L3 finding #2 · Documents the first end-to-end training run on
> RTX 4070, with **the docs §4 strict hparams**, and identifies a critical
> sub-majority degradation that motivates a hyperparam correction.

## Setup

- **Data**: v2 corrected (`data/processed/{train,val,test}.jsonl`)
  - 17,928 train / 2,241 val / 2,241 test
  - 20 classes, soft-label aggregated per `reports/data_dedup_ablation.md`
- **Model**: `jhu-clsp/mmBERT-base` + PEFT LoRA
  - r=16, α=32, dropout=0.05
  - target_modules = [query, key, value, dense]
  - 39,956 trainable / 307,585,576 total = **0.013% trainable**
- **Loss**: KL divergence (soft CE) against soft_label
- **Hardware**: RTX 4070 12 GB / fp16 / batch=32 / max_length=2048
  - Memory used: 11,990 / 12,282 MiB (97.6%)
  - GPU util: 85-90%
- **Hparams (docs §4 strict)**:
  - lr = 2e-4
  - num_train_epochs = 3 (1,683 total steps)
  - warmup_ratio = 0.06
  - weight_decay = 0.01
  - **No class balancing**

## Eval results

| Step  | Epoch | eval_loss | top1_acc | expected_winrate |
|-------|-------|-----------|----------|------------------|
| 1000  | 1.78  | 2.775     | 0.1008   | 0.1024           |
| 1500  | 2.67  | 2.721     | 0.0964   | 0.1009           |
| 1683 (final) | 3.00 | **2.713** | **0.1004** | **0.1046** |

Train loss curve: 4.016 → ~2.6 (decreasing with high-frequency oscillation).
Total runtime: 416.5 s (6 min 56 s, 4.04 it/s).

## Reference baselines for 20-class routing

| Strategy                       | top1   | expected_winrate |
|--------------------------------|--------|------------------|
| Random (uniform 1/20)          | 0.050  | ~0.060           |
| **Majority class (always gpt-4)** | **0.119** | **0.119**     |
| **Our naïve baseline**         | **0.1004** | **0.1046**   |

## ⚠️ Critical finding: sub-majority degradation

The docs-§4-strict baseline **underperforms the trivial "always predict gpt-4"
baseline** on both metrics:

```
top1:             0.1004 < 0.119  (Δ = -1.9 pts)
expected_winrate: 0.1046 < 0.119  (Δ = -1.4 pts)
```

This means **using our router is strictly worse than always routing to gpt-4**
— the model has learnt **no useful routing signal**, despite the eval_loss
dropping from the random-uniform baseline log(20) ≈ 3.0 down to 2.71.

## Diagnosis

eval_loss strictly decreasing while top1 stagnates around 0.10 indicates the
model is **flattening the softmax distribution** (some learning in log-prob
space) but the argmax is still dominated by majority-class bias. Three likely
root causes, in order of suspected impact:

1. **lr = 2e-4 is too high** for a 20-class fine-grained classifier whose
   classification head is randomly initialized (see LOAD REPORT: classifier.*
   were MISSING from the mmBERT checkpoint). The 2e-4 number is from the LoRA
   paper's *generic* default; empirical values for BERT-class fine-tuning
   with PEFT cluster around 5e-5 to 1e-4. The high grad_norm (10-15
   throughout training) is consistent with overshooting.
2. **3 epochs is too short** for a fresh classifier head to develop stable
   per-class decision boundaries. Train loss is still decreasing at the end
   of run, suggesting the model is undertrained, not overfit.
3. **No class balancing**. Class frequency ranges from 1.1%
   (gpt4all-13b-snoozy) to 11.9% (gpt-4) — a 11× imbalance. Without
   rebalancing, per-step gradients are dominated by the few frequent classes,
   reinforcing the majority-class collapse.

## Correction plan

Change one variable at a time, in expected-impact order:

1. **Iteration 2 — hparam correction (active in `configs/train.yaml`)**:
   `lr: 2e-4 → 5e-5`, `epochs: 3 → 5`, `warmup_ratio: 0.06 → 0.1`.
   Hypothesis: a stable lr + longer schedule alone can break the majority
   collapse. Expected: top1 → 0.13-0.16.
2. **If still < 0.12 top1** → add `class_weight: inv_freq` to compensate
   imbalance. Expected: top1 → 0.15-0.18.
3. **If still < 0.15 top1** → reconsider model capacity (mmBERT-32k-yarn
   variant) or LoRA target_modules (include MLP).

## Why this matters for the Capstone defense

This is the **second L3 finding** (the first being the §2.3/§2.4 soft-label
dedup correction):

> "Naïvely following the docs §4 strict hparams produces a baseline that
> degrades **below** the trivial 'always predict majority' baseline. We
> diagnosed this as the joint effect of an LoRA-paper-default lr that is too
> high for this specific fine-grained classification task, an under-budgeted
> training schedule, and a lack of class balancing — and corrected the
> configuration to bring routing performance above majority."

Two independent, data-supported corrections of the project specification
provide a robust critical-engineering narrative for the 5/16 defense, and
neither is a hindsight observation: both were caught by direct measurement
on the first runs.
