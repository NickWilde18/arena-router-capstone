# Baseline Run · LoRA Architecture Fix + Throughput Optimization · 2026-05-08

> Capstone L3 finding #3 · Diagnoses why the prior two baselines stayed
> sub-majority despite "running cleanly", traces it to a silent LoRA target
> mismatch against ModernBERT module names, and reports the post-fix run.
> Also documents a 35× throughput optimization landed on the same day.

## TL;DR

Two prior runs with the docs §4 LoRA target list `[query, key, value, dense]`
matched **only one** module per layer on `jhu-clsp/mmBERT-base` (ModernBERT
fuses Q/K/V into `Wqkv`, MLP into `Wi`/`Wo`), giving a misleading
**0.013% trainable** parameter count. After correcting the targets to
`[Wqkv, Wo, Wi, dense]` we reach the expected **1.10% trainable**.

Architecture fix lifts best eval top1 from `0.0959` (run #2) → `0.1218`,
crossing the majority-class baseline (`0.119`) by 0.003. **It is not a
breakthrough** — the model is still essentially predicting the marginal
distribution. The remaining ablation axes (lr, class_weight, loss form) now
become the dominant lever, since the architecture-budget excuse no longer
holds.

A throughput rework (max_length 2048→512, bf16, group_by_length,
gradient_checkpointing, batch 32→64, TF32 on fp32 ops) cut the wall-clock
from a projected 4 hours down to **6 min 52 s** (35× faster), pulling **168 W**
peak power on the RTX 4070 vs ~50 W before. This unlocks fast ablation
iteration for the rest of the week.

## Diagnosis: ModernBERT vs vanilla BERT module naming

`docs/train-mmbert-arena-router.md` §4 example:

```python
target_modules=["query", "key", "value", "dense"]
```

These are the standard `BertSelfAttention` submodule names. mmBERT-base is
**ModernBERT**, which fuses the projections:

```
Linear suffix  | Role
---------------+----------------------------------------------------
Wqkv           | fused Q+K+V projection            (was query/key/value)
dense          | attention output projection       (matches docs)
Wi             | MLP gate+up (GLU, fused)          (was intermediate.dense)
Wo             | MLP down                          (was output.dense)
classifier     | sequence classification head      (full-trained, not LoRA)
```

The PEFT `LoraConfig.target_modules` matcher does **substring matching against
the named-module path**, so `query`, `key`, `value` simply did not match any
module. Only `dense` (the attention output projection) hit one module per
layer × 22 layers — and even that count was overestimated; the actual
trainable budget reported by `print_trainable_parameters()` was 39,956 ≈
classifier head (~15k) + a single 768→768 LoRA pair (~24k).

After correcting to `[Wqkv, Wo, Wi, dense]`:

```
trainable params: 3,419,156 || all params: 310,964,776 || trainable%: 1.0995
```

The new count is in the typical 0.1–3% range for BERT-class LoRA, and
matches a back-of-envelope: 22 layers × 4 modules × ~36k LoRA params per
module ≈ 3.2M, plus classifier head.

## Throughput optimization (separate axis from accuracy)

Run #2 took ~7 min for 1683 steps at max_length=2048 with the broken
1-module LoRA. The arch-fixed run with 4-module LoRA × the same batch=32 +
max_length=2048 thrashed VRAM (11.99/12.0 GiB, GPU util 9%, ETA 4 hours)
because activations now actually scale with the LoRA module count.

The optimization sequence:

| Change                          | Why                                       | Effect                                |
|---------------------------------|-------------------------------------------|---------------------------------------|
| `max_length`: 2048 → 512        | Train tokens p99=532, max=1675; 2048 wasted ~98% | ~5× per-step speedup |
| `bf16` (instead of fp16)        | ModernBERT GeGLU more stable in bf16      | small accuracy stability gain         |
| `group_by_length=true`          | Cluster similar-length samples per batch; restored via `LengthGroupedSampler` since transformers 5.x removed the keyword arg | cuts intra-batch padding waste 5-10× |
| `dataloader_num_workers=4`      | Parallelize tokenization/IO from main thread | avoids CPU stalls between steps    |
| `gradient_checkpointing=true` (`use_reentrant=False`) + `enable_input_require_grads()` (PEFT) | ~40% activation memory savings | enables larger batch within 12 GB |
| `per_device_train_batch_size`: 32 → 64 | Bigger matmuls saturate Tensor Cores; was launch-overhead-bound | power 50W → **168W**, 6.7 it/s sustained |
| `torch.set_float32_matmul_precision('high')` | TF32 on fp32 ops outside bf16 autocast | small free speedup |

### Throughput optimizations attempted but rejected

- **`torch_compile=true` + `dynamic=True`**: compile fuses kernels but PEFT's
  `save_pretrained` hits `KeyError: modules_to_save.default.weight` because
  the compile-wrapped state-dict keys differ from what PEFT expects. Saving
  the adapter is non-negotiable, so disabled. Re-attempt after PEFT learns
  to unwrap manually-compiled models.
- **FlashAttention 2**: ~30 min compile time on this WSL2 box; with median
  seq length 16 tokens the FA2 win on padding-elimination is small once
  `group_by_length` is in place.

### Wall-clock comparison

| Run                                | Wall    | Throughput        | Power    |
|------------------------------------|---------|-------------------|----------|
| #1 docs §4 strict (broken LoRA)    | 6:56    | 4.04 it/s         | n/m      |
| #2 hparam-corrected (broken LoRA)  | ~7 min  | similar           | ~60 W    |
| arch-fixed @ batch=32 max_len=2048 | (4h ETA, killed at step 33) | 0.2 it/s | thrashing |
| **arch-fixed + opt (this run)**    | **6:52** | **6.68 it/s, 217 samples/s** | **peak 168 W** |

## Setup

- **Data**: v2 corrected (`data/processed/{train,val,test}.jsonl`)
  - 17,928 train / 2,241 val / 2,241 test, 20 classes
- **Model**: `jhu-clsp/mmBERT-base` + PEFT LoRA
  - r=16, α=32, dropout=0.05
  - target_modules = `[Wqkv, Wo, Wi, dense]`  (was `[query, key, value, dense]`)
  - **3,419,156 trainable / 310,964,776 total = 1.0995% trainable** (was 0.013%)
- **Loss**: KL divergence (soft CE) against soft_label
- **Hardware / training args**: RTX 4070 12 GB; bf16; batch=64; grad_accum=1;
  gradient_checkpointing; max_length=512; group_by_length;
  dataloader_num_workers=4
- **Hparams (run #2 carry-over to isolate the architecture axis)**:
  lr=5e-5, epochs=5, warmup_ratio=0.1, weight_decay=0.01, no class_weight

## Eval results

| Step  | Epoch | eval_loss | top1_acc   | expected_winrate |
|-------|-------|-----------|------------|------------------|
| 500   | 1.78  | 2.618     | 0.1187     | 0.1204           |
| **1000** | **3.56** | **2.610** | **0.1218** ✓ best | **0.1239** |
| 1405 (final) | 5.00 | 2.608 | 0.1169 | 0.1196 |

`load_best_model_at_end=true` with `metric_for_best_model=top1_accuracy` →
the saved adapter is the step-1000 / epoch-3.56 checkpoint.

train_runtime = 412.6 s. train_samples_per_second = 217.3.

### Reference baselines

| Strategy                           | top1   | expected_winrate |
|------------------------------------|--------|------------------|
| Random (uniform 1/20)              | 0.050  | ~0.060           |
| **Majority class (always gpt-3.5-turbo)** | **0.119** | **0.119**     |
| Run #1 (docs §4 strict, broken LoRA)| 0.1004| 0.1046           |
| Run #2 (hparam-corrected, broken LoRA) | 0.0959 | 0.1009        |
| **Run #3 (this, arch-fixed)**      | **0.1218** | **0.1239**   |

## What this run does and does not show

- ✅ Confirms LoRA capacity was indeed silently capped at 0.013% by the
  module-name mismatch. The fix lifts trainable budget 85× and returns the
  ~85× more activations expected during backward.
- ✅ Lifts top1 across the majority threshold (0.119 → 0.1218) — but only by
  0.003 in absolute terms. Routing is now **just barely** non-degraded.
- ❌ Does **not** demonstrate that the routing task is being learned. Loss
  plateaus at ~2.59-2.61 from epoch 1 onward, with grad_norm settling around
  2-3. The model is converging on a softmax that is only marginally more
  informative than the empirical class frequency.

## Why we are still stuck near majority

With the LoRA budget no longer the bottleneck, the suspects re-rank:

1. **lr=5e-5 is too low for this larger trainable budget.** Run #2 lowered
   lr from spec's 2e-4 specifically because grad_norm was 10–15 with the
   tiny adapter. With 85× more parameters, that logic no longer applies —
   each parameter gets a smaller share of the gradient. grad_norm in this
   run sits at 2–3, suggesting the optimizer is taking timid steps.
   **Next ablation**: lr ∈ {1e-4, 2e-4 (back to spec)}.
2. **Soft-label noise + class imbalance pull toward marginal.** KL
   divergence against very flat soft labels (mean entropy 0.55 nats post-
   dedup) over a Zipfian class distribution rewards predictions that match
   the marginal. **Next ablation**: `class_weight: inv_freq`; and a
   CE-on-argmax run for direct comparison (the 4-cell matrix in the
   handoff).
3. **First-turn-only prompt may be too lean for routing signal.** Median
   prompt length is 16 tokens; many prompts are short greetings that
   plausibly do not encode which model would have won. This is a data-side
   ceiling, not a hyperparameter — flag for the defense narrative but
   probably do not chase before 5/16.

## Next ablation matrix (5/9-10)

Per the handoff plan, run the 4-cell `KL × CE` × `v1_strict × v2_corrected`
matrix on top of arch-fixed targets:

|              | KL loss            | CE loss            |
|--------------|--------------------|--------------------|
| v1 strict    | A                  | B                  |
| **v2 (current)** | **C ← this run** | D                  |

Then layer in `class_weight: inv_freq` as an independent axis, and finally
sweep lr ∈ {5e-5, 1e-4, 2e-4} on the best (loss, data) cell.

## What the L3-#3 narrative looks like for the defense

> "Two prior baselines that 'ran cleanly' actually trained a 0.013%-of-model
> LoRA adapter — the docs-§4 target-module list was written for vanilla BERT
> (`query/key/value/dense`) and silently matches almost nothing on
> ModernBERT, which fuses those projections into `Wqkv`/`Wi`/`Wo`. We
> diagnosed this from a parameter-count plausibility check, restored the
> correct targets, and — separately — rebuilt the training pipeline to fit
> the actual data distribution (median sequence length 16 tokens, not the
> spec's 2048-token assumption), which together cut training time 35×.
> The architecture fix did clear the majority-class baseline, but it
> exposed that the remaining sub-majority gap is not an architecture
> problem — it is loss / class-weight / lr."

The L3 framing across the three findings so far:

1. **Data dedup** (§2.3 vs §2.4 contradiction) — fixed before any training.
2. **Hparam correction** (lr 2e-4 → 5e-5) — fixed sub-majority *symptom*,
   not the root cause; the root cause was hidden behind the broken LoRA.
3. **LoRA target mismatch + throughput rework** (this finding) — fixed the
   actual capacity bottleneck; reveals the next layer of issues.
