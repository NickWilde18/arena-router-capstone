# Data Deduplication Ablation · §2.3 / §2.4 Reconciliation

> Capstone L3 finding · 2026-05-07

## Problem: a hidden contradiction in the spec

The project specification (`docs/train-mmbert-arena-router.md`) contains an
internal contradiction between two adjacent sections:

- **§2.3 — Aggregate to soft labels.** Groups same-prompt records to produce a
  per-prompt distribution `P(m | prompt) ∈ Δ^N`, then trains with
  KL-divergence / soft cross-entropy. This **requires ≥2 records per prompt**
  to yield any non-trivial distribution.
- **§2.4 — Dedupe and filter.** Says: *"按小写+去空格后的 prompt hash 去近似
  重复"* — collapse rows whose normalized first-turn prompt hash collides.

Taken literally, §2.4 deletes precisely the multi-row groups §2.3 needs.
They cancel each other out.

## Empirical confirmation (v1 strict literal interpretation)

We first implemented §2.4 verbatim. Result on `train.jsonl` (17,238 rows):

| k = distinct models in `soft_label` | count | % |
|---|---|---|
| 1 (collapses to hard label)              | 15,070 | **87.4%** |
| 2 (only from `tie` battles → 0.5/0.5)    | 2,168  | 12.6% |
| ≥3 (genuine multi-model preference)      | **0**  | **0.0%** |

Mean Shannon entropy of `soft_label`: **0.126 bits**.

Therefore §3's example `{"gpt-4": 0.6, "claude-v1": 0.3, "vicuna-13b": 0.1}`
**never occurs**, and §4's `loss: kl_div` is mathematically equivalent to
plain cross-entropy on a 1-hot vector. There is no signal for a soft- vs
hard-label ablation.

## Correction

Replace the prompt-only dedup key with a tuple key:

| | dedup key | preserves |
|---|---|---|
| v1 (strict)    | `prompt_hash`                                     | one row per prompt |
| **v2 (corrected)** | `(prompt_hash, sorted(model_a, model_b))`     | same prompt under different model pairs |

Rationale: literal duplicate battles (same prompt, same pair, repeated
emission) are still removed — addressing §2.4's stated noise concern. But
the *cross-pair* observations of one prompt are kept, restoring §2.3's
intent.

## Empirical result (v2)

`train.jsonl` `soft_label` size distribution:

| k | v1 strict | v2 corrected | v2 / v1 |
|---|---|---|---|
| 1 | 15,070 (87.4%) | 14,457 (80.6%) | ×0.96 |
| 2 | 2,168  (12.6%) | 2,810  (15.7%) | ×1.30 |
| 3 | 0              | 387    (2.2%)  | — |
| 4 | 0              | 154    (0.9%)  | — |
| 5 | 0              | 62     (0.3%)  | — |
| 6-18 | 0           | 53             | — |

Mean Shannon entropy: **0.126 → 0.222 bits (+76%)**.

Filter retention:
- v1 strict     : 33,000 → 26,019 (78.8% kept)
- v2 corrected  : 33,000 → 31,933 (96.8% kept)

Genuine multi-model `soft_label` examples now appear:

```jsonl
{"text": "Where does \"nuchacho\" appear in fiction?",
 "soft_label": {"RWKV-4-Raven-14B": 0.33,
                "stablelm-tuned-alpha-7b": 0.33,
                "claude-v1": 0.33}}

{"text": "how many times does the letter \"e\" appear in \"ketchup?\"",
 "soft_label": {"gpt-4": 0.50, "RWKV-4-Raven-14B": 0.33, "vicuna-13b": 0.17}}

{"text": "You are provided the following categories which are descriptions of...",
 "soft_label": {"gpt-3.5-turbo": 0.43, "palm-2": 0.29,
                "alpaca-13b": 0.14, "fastchat-t5-3b": 0.14}}
```

## Why this matters for L3

The point of §4's `loss: kl_div` is to absorb judge noise via soft
distributions. Under v1 there is no distribution to absorb — KL ≡ CE on a
1-hot vector. Under v2:

1. **The KL-vs-CE ablation becomes meaningful.** 19.4% of train rows now
   carry non-degenerate distributional information; KL training has signal
   to exploit that CE cannot.
2. **`expected_winrate` (§4) stops being equivalent to top-1 accuracy.** The
   estimated win probability of the predicted model becomes interpolated
   over the soft label, giving a smoother evaluation surface.
3. **Per-prompt class-balanced loss weighting becomes well-defined.** Rare-
   model wins on contested prompts now get soft credit rather than being
   all-or-nothing.

## Reproducibility

- v1 strict baseline preserved at `data/processed/v1_strict/`
  (filtered parquet + 4 jsonl files + label2id.json).
- v2 corrected is the active output at `data/processed/`.
- To regenerate v1 from scratch, revert `src/data/filter.py` to commit
  `2ace959` and re-run `scripts/01_prepare_data.sh`.
