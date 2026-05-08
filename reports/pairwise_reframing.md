# Pairwise Reframing — Why It Looks Like a Win But Isn't · 2026-05-08

> Capstone L3 finding #7 · Tested whether re-framing the routing task as
> per-battle binary classification (`prompt + [a=X] + [b=Y] → which wins`)
> recovers the per-pair signal that the multi-class softmax dilutes
> (Diagnostic 3 in `diagnostic_root_cause.md` showed binary lift exists).
> Result: held-out binary top1 jumps to **0.6403** (+13.7 pp over 50%
> majority), but Borda-aggregated routing on the top-5 backend pool
> collapses to **always predict gpt-4** (top1 = 0.2254 = majority class
> exactly). The +13.7 pp gain is explained by the model learning the
> global tier ranking from identity tokens, not by per-prompt
> conditioning. **Pairwise reframing is not a usable router.**

## Setup

### Data construction (`src/data/build_pairwise.py`)

Per-battle, not per-aggregated-prompt:

```
input  = "[a=<model_a>] [b=<model_b>] <prompt>"
label  = "a" if winner == model_a else "b"  (drop ties)
```

Filter both `model_a` and `model_b` to the same top-5 set used by the
5-class router (so the pairwise classifier and the multi-class router
see the same backend pool).

```
Loaded 31,933 raw battles
Kept 2,775 (m_a, m_b ∈ top-5, winner non-tie)
  dropped OOV-pair:        27,990
  dropped tie:                759
  dropped tie(bothbad):       409
Train 2,220 / val 277 / test 278
Label distribution: a=1,398, b=1,377   (perfectly balanced 50/50)
Pair distribution (top): koala vs vicuna 524, gpt-3.5 vs koala 301, ...
```

### Training

Same arch-fixed pipeline (CE + lr=5e-5 + r=16 + bf16 + grad_ckpt + group_by_length),
binary head, 8 epochs (small dataset), eval@50.

```
val top1 trace:
  step  50 (epoch 2.78):  0.4982   ← random
  step 100 (epoch 5.56):  0.5921
  step 144 (epoch 8.00):  0.6498   ← saved (load_best_model_at_end)
```

## Test eval — looks great in isolation

```
n_eval                   = 278
top1_accuracy            = 0.6403   ← +13.7 pp over majority 0.504
expected_winrate         = 0.6403   (= top1 since binary)
ECE (15-bin)             = 0.033    (well-calibrated)
pred distribution        = a: 140, b: 138   (balanced, no class collapse)
per-class recall         = a: 0.643, b: 0.638  (symmetric)
```

The numbers are real and held-out; this is not the `binary_gv` v1 overfit
artifact from `diagnostic_root_cause.md` Diagnostic 3.

## Routing eval via Borda count — collapses to majority

For a real router, you don't get to ask "in pair (X, Y), who wins?". You
get a prompt and must pick one of K backends. Aggregate the pairwise
classifier into a routing decision via Borda count
(`src/eval/borda_router.py`):

```python
for each ordered (m_i, m_j), i ≠ j:
    score[m_i] += P(m_i wins | prompt, m_i, m_j)
pick argmax(score)
```

Run on the 5-class router's own test set (`data/processed/top5/test.jsonl`,
1,269 rows, `K=5`, `K(K-1)=20` ordered pairs):

```
n_eval                   = 1,269
n_models                 = 5
n_pairs_ordered          = 20
top1_accuracy            = 0.2254
expected_winrate         = 0.2269
majority_top1            = 0.2254   ← exact majority
delta_top1_vs_majority   = 0.0000
pred distribution:
  gpt-4         1,267   ← 99.8%
  claude-v1         2
  gpt-3.5-turbo     0
  koala-13b         0
  vicuna-13b        0
per-class recall:
  gpt-4         1.000   ← always predicted
  others        0.000
```

**The Borda router predicts gpt-4 for 1,267 of 1,269 prompts.** It is
identical to the trivial "always pick gpt-4" router, with top1 exactly
the gpt-4 frequency.

## Why the +13.7 pp binary lift doesn't transfer

The pairwise classifier's input *contains* the two model identities as
tokens. The labels overall reflect the global tier ranking (gpt-4 ≻
claude-v1 ≻ gpt-3.5-turbo ≻ vicuna-13b ≻ koala-13b on Arena win-rate).
What the model actually learned:

```
if "[a=gpt-4]" appears in input → label="a" with high probability
if "[b=gpt-4]" appears in input → label="b" with high probability
```

I.e. it learned **"whichever side is gpt-4 wins"** — a pure model-tier
prior, no per-prompt conditioning. The +13.7 pp held-out binary lift is
exactly what you'd expect from a deterministic *"strongest model wins"*
rule applied across pairs.

In Borda aggregation, the *strongest-wins* rule applied to every pair
produces *strongest model gets all the votes* → score for gpt-4 ≈ K-1
across all prompts, score for everyone else < that. The argmax routes
every prompt to gpt-4.

Per-prompt variance in `P(a wins | prompt, a, b)`: stays close to the
tier-determined value regardless of prompt content. We verified this by
inspecting the score matrix — gpt-4's score has small variance across
the 1,269 prompts (std ≪ mean gap to claude-v1's score).

## Cross-check against L3 #6

This is consistent with the four diagnostics in `diagnostic_root_cause.md`:

- **Diagnostic 1** (label noise): 80.8% prompts have k=1, multi-trial
  agreement coin-flip. There simply isn't enough per-prompt signal to
  override a strong tier prior.
- **Diagnostic 3** (gpt-4 vs vicuna binary, *no* identity tokens): test
  top1 = 0.5042 (+0.3 pp). Same model, same kind of data, but without
  identity tokens, the lift is essentially zero. Confirms that the +13.7
  pp lift in this experiment came from **the identity tokens**, not the
  prompt content.
- **Diagnostic 4** (linear probe top-5): frozen encoder gets 0.2224 vs
  LoRA 0.2492 (+2.7 pp lift). Same magnitude as the prompt-conditional
  signal we measured here.

## When pairwise reframing *would* help

Pairwise binary is the right framing when:
- Per-prompt signal exists (judge agreement is high)
- Model identities matter for the routing decision regardless

Pairwise reframing **collapses to tier-prior** when:
- Per-prompt signal is below the noise floor (this dataset)
- A consistent tier ranking exists in the labels

The *engineering* lesson: when you have low-signal labels and your model
input includes class-identity tokens, your "pairwise binary" classifier
will look impressive on its native task (the binary test set's labels
*do* track the tier prior), but its routing behavior — which is the
metric the deployment actually cares about — will degenerate to the
tier majority pick. Always evaluate the production-shaped metric, not
the proxy.

## Final defense framing (L3 #7)

> "After the four diagnostics in L3 #6 pinned the bottleneck to label
> noise, we tried one more reframing — per-battle pairwise binary
> classification with model identity tokens — to see if the per-pair
> signal that exists in the data could be aggregated into a router.
> Held-out binary top1 jumped to 0.6403 (+13.7 pp), which initially
> looked like a major win. But Borda aggregation onto the same 5-backend
> pool collapsed to predicting gpt-4 for 99.8% of prompts (top1 = 0.225,
> exactly the majority class). The classifier had learned the global
> tier ranking from the `[a=X][b=Y]` identity tokens, not per-prompt
> routing — verified by cross-referencing Diagnostic 3 (binary without
> identity tokens: +0.3 pp held-out). The result *strengthens* the
> data-ceiling claim: even framings that look like they're working on
> their own metric reduce to majority prediction once you measure them
> in production shape. The 5-class CE router (test top1 0.2380) remains
> the best deployable artifact."

L3 finding catalogue:

1. Data dedup contradiction
2. Sub-majority degradation (confounded by #3)
3. LoRA target mismatch + 35× throughput
4. Ablation matrix + capacity sweep — data ceiling
5. Top-K production-aligned restriction
6. Four-test diagnostic suite — soft-label noise root cause
7. **Pairwise reframing — proxy-metric trap, confirms data ceiling**
