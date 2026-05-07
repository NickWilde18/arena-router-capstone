# Ablation Matrix · Architecture-Fixed Baseline · 2026-05-08

> Capstone L3 finding #4 · Once the LoRA target mismatch was corrected
> (see `baseline_v2_kl_arch_fixed.md`), runs the orthogonal axes
> (lr / loss form / class_weight) on the same data + capacity. Reveals a
> hard ceiling at top1 ~0.125 — the model collapses to "predict vicuna-13b
> for ~60% of prompts" regardless of loss form. The plateau is **not**
> a hyperparameter problem; it is a data-side ceiling on the routing
> signal extractable from a first-turn prompt of median 16 tokens.

## Setup (constant across all cells)

- Data: v2 corrected (17,928 train / 2,241 val / 2,241 test, 20 classes)
- Model: `jhu-clsp/mmBERT-base` + LoRA on `[Wqkv, Wo, Wi, dense]`,
  r=16, α=32, dropout=0.05 → 1.10% trainable
- Optimizer: AdamW, weight_decay=0.01, 5 epochs, batch=64
  (effective via grad_accum=1), bf16, gradient_checkpointing
- Eval cadence: every 500 steps; `load_best_model_at_end=true`,
  metric=`top1_accuracy`

The handoff plan called for a 4-cell `KL × CE` × `v1_strict × v2_corrected`
matrix. The v1 strict snapshot was not preserved on disk in this clone
(`data/processed/` only has the current v2 split + originals), so only the
v2 row is reported. The KL × CE comparison still isolates the loss-form
axis cleanly; the v1 vs v2 axis was already covered in
`reports/data_dedup_ablation.md` (L3 #1).

## Ablation cells run

All numbers are **best val top1 across in-run evals** (the checkpoint that
`load_best_model_at_end` selects). All cells share LoRA `r=16, α=32` unless
noted; capacity sweep is in its own table below.

| #   | loss          | lr     | warmup | class_weight | best val top1 | best val exp_winrate | best epoch | notes |
|-----|---------------|--------|--------|--------------|---------------|----------------------|------------|-------|
| #2 baseline (carried over) | KL  | 5e-5  | 0.10  | none      | 0.1218 | 0.1239 | 3.56 | run #3, the post-arch-fix anchor |
| L1  | KL  | 1e-4  | 0.10  | none      | 0.1214 | 0.1244 | 3.56 | top1 frozen at .1214 across all evals |
| L2  | KL  | 2e-4  | 0.06  | none      | 0.1156 | 0.1164 | 1.78 | overfit; .1017 at ep 3.56, .0946 at ep 5 |
| C1  | KL  | 5e-5  | 0.10  | inv_freq  | **0.0558** | 0.0537 | (aborted at ep 1.78) | rare-class collapse; killed |
| **CE** | CE | 5e-5  | 0.10  | none      | **0.1254** | 0.1264 | 3.56 | best non-capacity cell; saved as final |
| CE+IF | CE | 5e-5  | 0.10  | inv_freq  | 0.0348 | 0.0321 | (aborted at ep 1.78) | also rare-class collapse; killed |

### Capacity ablation (CE + lr=5e-5, vary LoRA r)

| LoRA r | LoRA α | trainable params | trainable % | best val top1 | best val exp_winrate |
|--------|--------|------------------|-------------|---------------|----------------------|
| 16     | 32     | 3,419,156        | 1.10%       | 0.1254        | 0.1264               |
| 32     | 64     | 6,822,932        | 2.17%       | 0.1258        | 0.1272               |
| 64     | 128    | 13,630,484       | 4.24%       | 0.1258        | 0.1292               |

Doubling and quadrupling the adapter budget moves val top1 by 0.0004 — one
sample of noise. **Capacity is not the bottleneck.**

Reference baselines:

| Strategy            | val top1 | comment                            |
|---------------------|----------|------------------------------------|
| Random (uniform)    | 0.050    | 1/20                               |
| Majority class      | 0.119    | always predict gpt-4 (true majority on val) |
| **Best adapter (CE)** | **0.1254** | +0.006 over majority             |

## Read across the matrix

**lr is not the bottleneck.** Three lr settings (5e-5, 1e-4, 2e-4) all land
in 0.116–0.122 — same noise band. lr=2e-4 (the docs §4 spec) overshoots;
it gets the lowest train_loss (2.55 vs 2.65 for 5e-5) but val top1 falls
to 0.0946 by epoch 5 — classic overfit signature. lr=5e-5 (run #2's
choice, kept here for arch-axis isolation) is also empirically the best
for this larger LoRA budget. **Take-away: run #2's lr decision turned out
to be correct for the wrong reason** — they lowered lr because the broken
LoRA had high grad_norm; with the actual budget, grad_norm drops to 2-3
and 5e-5 is still the right call.

**Loss form: CE > KL by a small margin.** CE wins by 0.0036 top1
(0.1254 vs 0.1218) and shifts predictions slightly back toward the true
majority (gpt-4 from 23% → 35% of predictions). The small gap means
**the soft-label signal is not the main driver of accuracy here** — the
"smarter" KL loss against soft targets buys very little, because the
soft labels themselves are dominated by mode-collapse on vicuna.

**class_weight=inv_freq destroys accuracy regardless of loss form.**
KL+inv_freq → top1=0.056. CE+inv_freq → top1=0.035. Both worse than
random (0.050). The 11× weight range (min 0.256 for gpt-4, max 2.798 for
gpt4all-13b-snoozy) pushes per-step gradients on rare-class samples up
disproportionately; the model learns to predict rare classes whose actual
test-time priors are tiny, so top1 collapses below random.

A weaker reweighting (e.g. `sqrt(inv_freq)`, capped 1×–2× range) might
work better, but is unlikely to break the underlying bottleneck — see below.

## Diagnostic: vicuna-13b mode collapse

Running the saved adapters on the val set with full prediction histograms
(via `src/eval/metrics.py`):

```
KL (run #3) val: predicts vicuna-13b 1526/2241 = 68.1%
CE          val: predicts vicuna-13b 1297/2241 = 57.9%
                 predicts gpt-4       776/2241 = 34.6%
                 predicts other 18 classes ≤ 5% combined
```

Both runs converge on a 2-class output distribution `{vicuna-13b, gpt-4}`
plus trace amounts of 2 other classes. **14 of the 20 classes are never
predicted** by either adapter on the val set.

Why vicuna and not the actual majority gpt-4? With KL on soft labels,
"tie" battles split weight 0.5 / 0.5 between the two participating models.
Vicuna-13b participates in many ties (it was a frequent Arena contestant
mid-2023), so per-prompt soft mass on vicuna stacks up across the training
set. The KL gradient pushes the model to put softmax mass where the
target's mass is — and vicuna ends up with the highest *integrated* soft
mass, even though it is not the most frequent argmax. CE strips out the
soft mass and uses only the per-prompt argmax, which partially restores
gpt-4 — but not enough, because the underlying training signal is still
weak.

## Why the ceiling is data-shaped, not lr/loss-shaped

Three converging signals say the bottleneck is upstream of training:

1. **Median prompt length is 16 tokens.** Many train prompts are short
   greetings or one-line questions ("hi", "what is X") whose answer
   quality genuinely does not depend on which model produced it. There
   is no routing signal to learn from these.
2. **Per-class recall is bimodal.** The two predicted classes
   (vicuna-13b, gpt-4) have non-trivial recall (0.58, 0.38). All other
   18 classes have recall ≈ 0. The model is not making "soft" mistakes —
   it is collapsing onto the few classes that the noisy soft labels make
   look attractive.
3. **eval_loss is 2.61 nats but log(20) = 3.0.** The model is only barely
   below the uniform-prediction baseline in CE-of-soft-target space. It
   has learned *some* averaged class prior, almost no per-prompt
   conditional structure.

This is consistent with the Arena soft-label noise level: mean target
entropy 0.55 nats post-dedup (from `data_dedup_ablation.md`). With per-
prompt entropy that high, the optimal Bayes classifier is itself close to
the marginal — beating majority by 0.5 pp is roughly what is achievable.

## Test set evaluation on the best (CE) adapter

```
n_eval:                 2241
top1_accuracy:          0.1232
expected_winrate:       0.1199
ECE (15-bin):           0.0171
majority_top1 (gpt-4):  0.1187
delta vs majority:      +0.0045
```

Test top1 = 0.1232, almost identical to val (0.1254) → **no overfitting to
val**, the small win over majority generalizes. ECE 0.017 means the
predicted probabilities are well-calibrated (a 1.7% gap between confidence
and accuracy across confidence bins).

Pred distribution on test mirrors val: 1304 predictions on vicuna-13b
(58%), 768 on gpt-4 (34%), 14 classes never predicted. Same mode collapse.

## Decision and next steps

**Best configuration locked**: CE loss + lr=5e-5 + warmup=0.1 + 5 epochs +
no class weighting. Saved at `outputs/ablation_ce_loss/final/`.

What is **worth** trying before the defense (5/16):

- **Conversation-context features** — concatenate `model_a` ID or first
  assistant turn into the input. Cheapest oracle baseline: include the
  winner's identity in the prompt; if oracle hits ≈1.0 we know the
  encoder *can* learn the mapping when given an informative input, which
  isolates the bottleneck cleanly to "first-turn prompt is too lean".
- **Output-side temperature calibration** — does not change top1 but may
  improve ECE / expected_winrate.

What is **not worth** chasing (already ruled out by ablation):

- **lr / warmup / weight_decay** — three lr settings span 0.116-0.122,
  same noise band.
- **class_weight** — both inv_freq attempts collapsed below random.
- **LoRA capacity** — r=16/32/64 all land at top1 ≈ 0.1255 ± 0.0004.
- **Longer training** — eval top1 peaks at epoch 3.56 and decays;
  extending epochs just overfits.

## Defense framing (L3 #4)

> "After fixing the LoRA target mismatch (L3 #3), we ran four orthogonal
> axes — three learning rates, two loss forms, two class-weighting
> schemes, three LoRA ranks (r=16/32/64) — and found a hard ceiling at
> val top1 ≈ 0.125 vs majority 0.119. Diagnostic eval on saved adapters
> revealed the bottleneck: the model collapses to predicting vicuna-13b
> for 58–68% of prompts and gpt-4 for most of the rest, never predicting
> 14 of the 20 classes. The collapse traces to the soft-label tie-mass
> distribution, which CE partially mitigates and `inv_freq` cannot help
> with — but the deeper reason is that the median Arena first-turn
> prompt has 16 tokens, too few to encode which model would have won.
> Quadrupling LoRA capacity (r=16 → r=64) moved top1 by 0.0004 = one
> sample. The remaining 0.5pp gap over majority is most plausibly the
> Bayes-optimal margin given the data, not a hyperparameter we have left
> on the table."

L3 findings to date:

1. **Data dedup** (§2.3 / §2.4 contradiction)
2. **Hparam correction (lr 2e-4 → 5e-5)** — fixed the *symptom* of the
   broken LoRA; was right for the wrong reason
3. **LoRA target mismatch + 35× throughput rework** — fixed the actual
   capacity bottleneck
4. **Data-side ceiling (this finding)** — the remaining sub-+1pp gap
   over majority is upstream of the model, not the optimizer
