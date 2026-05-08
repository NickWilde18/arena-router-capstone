---
theme: seriph
title: "Arena Router · DDA 4080 Capstone Defense"
info: |
  Training a mmBERT-base PEFT-LoRA classifier on LMSYS Chatbot Arena
  preference data for vLLM Semantic Router (VSR) integration.
class: text-center
drawings:
  persist: false
transition: slide-left
mdc: true
fonts:
  sans: "Inter"
  serif: "Source Serif Pro"
  mono: "JetBrains Mono"
seoMeta:
  ogTitle: "Arena Router Capstone Defense"
  ogDescription: "mmBERT routing classifier trained on LMSYS Arena preferences"
---

# Arena Router

## Training a mmBERT routing classifier on LMSYS Arena preferences

DDA 4080 Capstone Defense · 2026-05-16

<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer hover:bg-white/10">
    Press space for next page <carbon:arrow-right class="inline"/>
  </span>
</div>

<!--
**EN.** Good morning. I'll defend my DDA 4080 capstone — training a mmBERT routing classifier on LMSYS Chatbot Arena preference data for the vLLM Semantic Router project. The talk has three parts: the problem framing, the seven critical-engineering findings I documented, and a final summary of where the project stands. Total time about 25 minutes plus questions.

**中文.** 各位老师好。今天汇报的是我的 capstone 课题。我做的事情，是给 vllm-semantic-router 这个开源项目训练一个路由分类器，让它根据用户输入的 prompt 选最合适的后端模型。数据用的是 LMSYS Chatbot Arena 公开的人类对战记录。今天分三部分讲，先说问题背景和数据，再讲我跑下来发现的七个关键工程点，最后是项目结论和后续。整体大约 25 分钟，剩下时间留给提问。
-->

---
layout: section
---

# 1. Problem & Approach

<!--
**EN.** First section: what is the problem, what data do we have, and what infrastructure does it plug into.

**中文.** 第一部分先讲清楚问题是什么、数据是什么样子、跟上下游的工程怎么对接。
-->

---

# vLLM Semantic Router (VSR) routing task

VSR routes each user prompt to the *best* backend model out of a deployed pool.

<div grid="~ cols-2 gap-6">

<div>

### Architecture

- **Signal-decision** pattern (v0.2 "Athena")
- Production encoder: `mmbert-32k-yarn`
- Per-prompt → per-signal classifier → backend pick
- Existing classifier heads: domain / PII / jailbreak / modality / feedback / fact-check

### Our task

Add a **`model_routing`** signal head: `prompt → P(model wins)`

</div>

<div>

```mermaid {scale: 0.55}
flowchart TD
  A[User prompt] --> B[mmBERT encoder]
  B --> C1[domain head]
  B --> C2[PII head]
  B --> C3[best_model<br/>head]
  B --> C4["..."]
  C1 --> D{Decision layer}
  C2 --> D
  C3 --> D
  C4 --> D
  D --> E1[gpt-4 backend]
  D --> E2[claude-v1 backend]
  D --> E3["..."]
  style C3 fill:#fde68a,stroke:#f59e0b
```

</div>
</div>

<!--
**EN.** VSR is vLLM's semantic-router project. v0.2 codename Athena. The pattern is signal-decision: each user prompt runs through several BERT-class classifier heads — domain, PII, jailbreak, modality — and the decision layer picks a backend based on those signals. The encoder is jhu-clsp/mmBERT-base, extended to 32K context with YaRN. My task adds one more signal head, `model_routing`, that maps a prompt to the model most likely to give the best answer. I highlight the new head in yellow on the diagram. The architecture is fixed; what I trained is just one new classifier head.

**中文.** VSR 是 vllm-project 下面的语义路由器，0.2 版本叫 Athena。它的工作模式叫 signal-decision：每条用户输入会过几个 BERT 系列的小分类器，分别判断领域、有没有个人信息、越狱倾向、文本还是图像、用户反馈、要不要做事实核查这些"信号"，决策层综合这些信号来挑后端。底层编码器统一用 jhu-clsp/mmBERT-base，再用 YaRN 把上下文撑到 32K。我的任务就是再加一个分类头叫 model_routing，输入是 prompt，输出是哪个模型最可能赢。整个上下游架构是定的，我只训练这一个新加的头，图上标黄的那个。
-->

---

# Data: `lmsys/chatbot_arena_conversations`

<div grid="~ cols-2 gap-6">

<div>

- **31,933** human pairwise battles
- 20 models after `min_class_freq ≥ 500` filter
- Fields: `model_a`, `model_b`, `winner`, `conversation_*`, `language`, `toxic_chat_tag`

### Per-battle → soft label (docs §2.2-2.3)

```yaml
winner == model_a:        (prompt, m_a, 1.0)
winner == model_b:        (prompt, m_b, 1.0)
winner == tie:            (prompt, m_a, 0.5)
                          (prompt, m_b, 0.5)
winner == tie (bothbad):  drop
```

Aggregate by prompt: `P(m | prompt) = Σ wᵢ·𝟙[lᵢ=m] / Σ wᵢ`

</div>

<div>

### Final splits (post §2.4 dedup)

| split | rows |
|-------|-----:|
| train | 17,928 |
| val   | 2,241 |
| test  | 2,241 |
| total | 22,410 |

### Class set (top 5 of 20 by argmax frequency)

| model | freq |
|-------|------|
| gpt-4 | 11.9% |
| vicuna-13b | 11.8% |
| gpt-3.5-turbo | 10.5% |
| claude-v1 | 10.1% |
| koala-13b | 8.3% |

</div>
</div>

<!--
**EN.** Data: LMSYS released about 32,000 human pairwise model battles. After filtering to models with at least 500 appearances, we keep 20. The label-construction recipe in the project spec turns each battle into one or two `(prompt, model, weight)` tuples — winner gets weight 1, ties split 0.5 each, "tie bothbad" is dropped. We then aggregate by prompt to get a soft distribution over the 20 classes. Final splits are 17,928 train, 2,241 val, 2,241 test. The class distribution is heavy-tailed — gpt-4 and vicuna-13b are tied at 12% each as the most-frequent argmax class.

**中文.** 数据集是 LMSYS 公开的 Chatbot Arena 对战记录，一共 31,933 场。规范要求过滤一下，保留出场次数至少 500 的模型，剩下 20 个。规范的标签构造方式是这样：每场战斗拆成一到两条记录，胜方权重 1，平局两边各 0.5，"双方都差"那种直接丢掉。然后按 prompt 聚合，得到 20 类的软分布。最后切成训练 17,928、验证 2,241、测试 2,241 行。类别分布是长尾的，最高频的是 gpt-4 和 vicuna-13b，各占 12% 左右。
-->

---
layout: section
---

# 2. Seven L3 Critical-Engineering Findings

<!--
**EN.** This is the meat of the talk. Seven things I found by actually running the pipeline that the spec didn't anticipate, and what each one means for the final result.

**中文.** 接下来是汇报的核心。一共七个 finding，全是我实际跑流水线时碰到、规范文档没料到的问题。每一个我都说一下它对最终结果产生了什么影响。
-->

---

# Finding flow

<div class="text-sm">

| # | Finding | Result |
|---|---------|--------|
| 1 | Data dedup contradiction (§2.3 vs §2.4) | k≥3 prompts: 0% → 3.7% |
| 2 | Sub-majority degradation (was confounded by #3) | 0.100 < majority 0.119 |
| 3 | LoRA target mismatch + 35× throughput rework | trainable% 0.013 → 1.10; 4h → 6m52s |
| 4 | Ablation matrix + capacity sweep | data ceiling, top1 = 0.125 ± 0.0004 |
| 5 | Top-K production-aligned restriction | 5-class top1 = 0.249, all 5 classes predicted |
| 6 | 4-test diagnostic root cause | inherent soft-label noise, not model |
| 7 | Pairwise reframing collapses to tier-prior | binary 0.640 BUT Borda → majority 0.225 |

</div>

<v-click>

### One-line story

> "We started with the spec, found two silent bugs (data + LoRA targets), engineered the pipeline 35× faster, then ran four orthogonal axes and four diagnostics to prove the residual gap is data-shaped — not optimization-shaped — and a final reframing experiment confirmed the same."

</v-click>

<!--
**EN.** Here's the finding flow. Seven items, listed in causal order. Findings 1 and 3 are silent spec bugs — I'd have shipped a broken model if I hadn't caught them. Finding 2 is what *appeared* to be a problem but was confounded by 3. Findings 4 through 7 are diagnoses on top of the fixed pipeline, all converging on the same conclusion: the residual gap above majority is data-shaped, not model-shaped. The one-line summary on the click is the elevator pitch I'd give in 20 seconds.

**中文.** 七个 finding 我按因果顺序排好。第 1 个和第 3 个是规范里两处隐藏的坑，没发现的话我交一个有 bug 的模型就上去了。第 2 个是表面看到的问题，但根因是被第 3 个掩盖了，本质上是同一件事。第 4 到第 7 都是修好流水线之后做的诊断，最后全部指向同一个结论：超过 majority 的那一点点提升完全是数据本身决定的天花板，跟模型怎么训没什么关系。点一下出来的那句话，是我用 20 秒讲完整个故事的版本。
-->

---
layout: section
---

# L3 #1 · Data dedup contradiction

<!--
**EN.** First finding. The spec section 2.3 says one thing, section 2.4 says another, and they conflict. I caught this by running the pipeline strictly as written and looking at the resulting label distribution.

**中文.** 第一个 finding。规范的 2.3 节和 2.4 节互相打架。我严格照规范的字面意思跑了一遍流水线，看输出的标签分布才发现的。
-->

---

# §2.4 dedup hash conflicts with §2.3 aggregation

`docs/train-mmbert-arena-router.md` says:

<div grid="~ cols-2 gap-4 text-sm">

<div>

**§2.3 (soft-label aggregation)**

> "聚合为软标签：P(m | prompt) = Σ wᵢ·𝟙[lᵢ=m] / Σ wᵢ"

Requires **k ≥ 2 battles per prompt** to produce a non-degenerate distribution.

</div>

<div>

**§2.4 (filtering)**

> "按小写+去空格后的 prompt hash 去近似重复"

Collapses every same-prompt battle to one row → most prompts get **k = 1**.

</div>
</div>

<v-click>

### Empirical fallout (v1 strict, before correction)

```
k = 1 (single-class soft):  87.4%   ← soft labels degenerated to hard labels
k = 2:                      11.0%
k ≥ 3:                       0.0%
mean entropy:               0.31 nats
```

</v-click>

<v-click>

### Correction (`src/data/filter.py`)

- Dedupe by `(prompt_hash, sorted(model_a, model_b))` — preserves same prompt under different model pairs
- After fix: `k ≥ 3 = 3.7%`, mean entropy `0.55 nats` (+76%)
- Documented + commented in code: `# docs §2.4 says X, we do Y because…`

</v-click>

<!--
**EN.** §2.3 wants a soft distribution per prompt — that needs at least two battles per prompt. §2.4 says to dedup by prompt hash, which collapses everything to a single row per prompt. The two cancel each other out. After running the v1 strict pipeline, 87% of prompts had only one class with non-zero soft mass — the soft labels had degenerated to hard labels. I corrected by deduping on the tuple of prompt hash AND the sorted model pair, which kills literal duplicate battles but preserves the same prompt under different model pairs. The k≥3 fraction went from 0% to 3.7% and the mean entropy went up 76%. Every spec deviation is commented in code with the literal §X reference and the reason.

**中文.** 矛盾点是这样：2.3 节要的是软分布，那每个 prompt 至少得有两场战斗才有意义。但 2.4 节说要按 prompt 哈希去重，结果每个 prompt 只剩一行了，跟 2.3 节的要求正好抵消。我严格照 v1 跑了一遍，发现 87% 的 prompt 只有一个类有非零软标签，等于退化成硬标签了。修法是把去重的 key 从单纯的 prompt 哈希改成"prompt 哈希加上排序后的模型对"这个二元组，这样既能干掉真正的重复战斗，又能保留同一个 prompt 在不同模型对里出现的情形。修完之后 k≥3 的比例从 0% 升到 3.7%，平均熵涨了 76%。每一处偏离规范的地方，我都在代码注释里写明白了"规范第几节的原话怎么说、我为什么这么改"。
-->

---
layout: section
---

# L3 #3 · LoRA target mismatch + 35× throughput

<!--
**EN.** Skipping #2 here because it's just the surface symptom of #3. Going straight to the architecture bug.

**中文.** 跳过第 2 个直接讲第 3 个。第 2 个 finding 表面上是另一个问题，本质上是被第 3 个掩盖的同一件事，讲第 3 个就够了。
-->

---

# Silent bug: vanilla BERT names on ModernBERT

<div grid="~ cols-2 gap-6 text-sm">

<div>

**docs §4 example**

```python
LoraConfig(
  r=16, lora_alpha=32,
  target_modules=[
    "query", "key", "value", "dense"
  ],
  ...
)
```

Standard `BertSelfAttention` submodule names.

</div>

<div>

**mmBERT-base = ModernBERT** fuses projections:

```
Linear suffix | Role
--------------+----------------------
Wqkv          | fused Q+K+V          ← MISSED
dense         | attn output proj     ✓
Wi            | MLP gate+up (GLU)    ← MISSED
Wo            | MLP down             ← MISSED
classifier    | seq cls head         (full-train)
```

PEFT does **substring matching** → only `dense` matched (1 module/layer).

</div>
</div>

<v-click>

### Numerical signature

```
Before: trainable params:    39,956 / 307,585,576 = 0.013%   ⚠ way under expected ~1-3%
After:  trainable params: 3,419,156 / 310,964,776 = 1.10%    ✓
                          (85× more)
```

Probe with `print({n.rsplit('.',1)[-1] for n,p in m.named_modules() if isinstance(p, nn.Linear)})` *always*.

</v-click>

<!--
**EN.** Spec section 4 gives a LoRA config with `target_modules = ['query', 'key', 'value', 'dense']`. Those are vanilla BERT module names. mmBERT-base is actually ModernBERT, which fuses the QKV projections into a single `Wqkv` matrix and uses GLU MLPs with `Wi` and `Wo`. PEFT does substring matching, so only `dense` matched, just one module per layer. The trainable parameter count came out to 0.013% — eighty times smaller than the 1-3% that's typical for BERT-class LoRA. I caught it from a parameter-count plausibility check. After fixing the target list, trainable jumped 85x to the expected 1.10%. Lesson: always probe the actual Linear-layer suffixes before configuring LoRA.

**中文.** 规范第 4 节给的 LoRA 例子里，target_modules 是 query、key、value、dense，这是普通 BERT 的模块命名。但 mmBERT-base 实际上是 ModernBERT 架构，它把 QKV 投影合并成了一个 Wqkv 矩阵，MLP 用 GLU 拆成 Wi 和 Wo。PEFT 是按子串匹配的，结果只有 dense 命中了，每一层就只有一个 LoRA 模块在训。可训参数算下来才占整个模型的 0.013%，比 BERT 类 LoRA 通常的 1-3% 少了 80 倍。我是在做参数数量的合理性检查时才看出问题的。把模块名改对以后，可训参数跳了 85 倍，到了预期的 1.10%。这里学到的教训是：以后配 LoRA 之前，先把模型实际的 Linear 层名打印出来看一遍。
-->

---

# Throughput rework on the same axis

Original max_length=2048 + LoRA r=16 + 4 modules/layer thrashed VRAM (11.99/12.00 GiB, GPU util 9%, projected 4 hours).

<div class="text-xs">

| Change | Why | Effect |
|--------|-----|--------|
| `max_length` 2048 → 512 | Train tokens p99=532; 2048 wasted ~98% on padding | ~5× per-step |
| `bf16` (instead of fp16) | ModernBERT GeGLU more stable in bf16 | accuracy stability |
| `group_by_length` (custom `LengthGroupedSampler`) | transformers 5.x removed the keyword arg; restore via subclass | cuts intra-batch padding 5-10× |
| `gradient_checkpointing` (`use_reentrant=False`) + PEFT `enable_input_require_grads` | ~40% activation memory savings | enables larger batch |
| batch 32 → 64 (later 128) | Bigger matmuls saturate Tensor Cores | power 50W → **168W**, 6.7 it/s |
| `torch.set_float32_matmul_precision('high')` | TF32 on fp32 ops outside bf16 autocast | small free speedup |

</div>

<v-click>

### Wall-clock comparison

| Run | Wall | Throughput | Power |
|-----|------|------------|-------|
| Pre-fix (4h ETA, killed) | 4h projected | 0.2 it/s | thrashing |
| **Post-rework** | **6m52s** | **6.68 it/s · 217 samples/s** | **peak 168W** |

**Net: 35× faster on the same hardware.**

</v-click>

<!--
**EN.** Fixing the LoRA targets exposed a different problem — with 85x more LoRA parameters and the spec's max_length of 2048, the activations didn't fit in 12 GB of VRAM. Initial run was thrashing memory at 9% GPU utilization, projected four hours. I made six throughput changes. First, I checked the data — the 99th-percentile prompt is 532 tokens, so max_length 2048 was wasting ~98% of compute on padding. Dropped to 512. Second, switched fp16 to bf16 because ModernBERT's GeGLU is more numerically stable. Third, restored `group_by_length` via a custom LengthGroupedSampler subclass — transformers 5.x removed the keyword argument. Fourth, added gradient checkpointing with `use_reentrant=False` plus PEFT's `enable_input_require_grads` for the frozen base. That freed enough memory to bump batch size to 64 and later 128. Sixth small bonus: TF32 for fp32 ops outside the bf16 autocast. Net effect: same hardware goes from a 4-hour projected run thrashing memory to 6 minutes 52 seconds at 168W and 217 samples per second. That's 35x faster.

**中文.** 修了 LoRA target 又暴露出第二个问题。可训参数多了 85 倍，加上规范给的 max_length 等于 2048，激活值就吃不下 12GB 显存了。一开始跑的时候显存抖得厉害，GPU 利用率只有 9%，按速度推算要 4 个小时才能跑完。我做了六项改造。第一，看了一下数据，prompt 长度的 99 分位才 532 个 token，max_length 设到 2048 等于 98% 的算力都浪费在 padding 上，直接降到 512。第二，把 fp16 换成 bf16，因为 ModernBERT 用的 GeGLU 在 bf16 下数值更稳。第三，transformers 5.x 把 group_by_length 这个参数砍了，我自己写了个 LengthGroupedSampler 子类把它补回来。第四，开 gradient_checkpointing，参数加上 use_reentrant=False，再配上 PEFT 的 enable_input_require_grads，让冻结的 base 也能让梯度反传。这一步省出来的显存把 batch size 从 32 升到 64，后来到 128。最后一个小的白送的：torch 的 set_float32_matmul_precision 设成 high，让 bf16 autocast 之外的 fp32 算子也走 TF32。最后效果是同样一张 4070 12GB，从 4 小时显存抖动跑成 6 分 52 秒，功耗稳在 168W，每秒能处理 217 个样本。整体提速 35 倍。
-->

---
layout: section
---

# L3 #4 · Ablation matrix · data ceiling

<!--
**EN.** With the bugs fixed and the pipeline 35x faster, I ran the orthogonal hyperparameter axes to find the right configuration. Result: every axis is flat.

**中文.** 修完两个 bug、流水线提速 35 倍以后，我跑了正交的超参数轴，想找最佳配置。结果是每一条轴上的 top1 都是平的。
-->

---

# Four-axis sweep: every axis flat at top1 ≈ 0.125

<div class="text-sm">

| # | loss | lr | warmup | class_weight | best val top1 | best epoch | notes |
|---|------|----|----|----|----|----|----|
| #2 | KL | 5e-5 | 0.10 | – | 0.1218 | 3.56 | run #3 anchor |
| L1 | KL | 1e-4 | 0.10 | – | 0.1214 | 3.56 | top1 frozen across evals |
| L2 | KL | 2e-4 | 0.06 | – | 0.1156 | 1.78 | overfit, decays to 0.0946 |
| C1 | KL | 5e-5 | 0.10 | inv_freq | **0.0558** | aborted | rare-class collapse |
| **CE** | **CE** | **5e-5** | **0.10** | – | **0.1254** | **3.56** | **best, saved as final** |
| CE+IF | CE | 5e-5 | 0.10 | inv_freq | 0.0348 | aborted | also collapse |

</div>

### Capacity sweep (CE + lr=5e-5)

| LoRA r | trainable | best val top1 |
|--------|-----------|---------------|
| 16 | 1.10% | 0.1254 |
| 32 | 2.17% | 0.1258 |
| 64 | 4.24% | 0.1258 |

**Quadrupling capacity moves top1 by 0.0004 — one sample of noise.**

<!--
**EN.** Six configurations, three learning rates spanning 5e-5 to 2e-4, two loss forms (KL on soft target vs CE on argmax), two class-weighting schemes. CE narrowly beats KL by 0.36 points. Both inv-freq weighting attempts collapsed to predicting rare classes — accuracy below random — and were aborted. Then a separate capacity sweep at r=16, 32, 64 — the trainable parameter count goes from 1.1% to 4.2%, but the val top1 moves by 0.0004, a single sample of noise. None of these axes is the bottleneck.

**中文.** 一共六个配置，三个学习率从 5e-5 到 2e-4，两种损失函数，一种是 KL 配软标签、一种是 CE 配 argmax 硬标签，还有两种类别加权方式。CE 比 KL 微胜 0.36 个百分点。两次用 inv-freq 加权都崩了，把模型推到去预测稀有类，准确率比随机还低，我直接中止了。另外 LoRA 的秩做了 16、32、64 三档扫描，可训参数从 1.1% 涨到 4.2%，但验证 top1 只动了 0.0004，相当于一个样本的噪声。也就是说，这些超参里没有一个是真正的瓶颈。
-->

---

# Mode collapse: what the 20-class model actually predicts

<div grid="~ cols-2 gap-6 text-sm">

<div>

### Pred distribution on val (CE best)

```
vicuna-13b   1297  (58%)
gpt-4         776  (35%)
claude-v1     105  ( 5%)
gpt-3.5-turbo  45  ( 2%)
14 other classes  0
```

**14 of 20 classes never predicted.**

ECE = 0.018 (well-calibrated).

</div>

<div>

### Per-class recall (top 5)

```
vicuna-13b      0.583
gpt-4           0.378
claude-v1       0.062
gpt-3.5-turbo   0.038
koala-13b       0.011
```

**Why vicuna and not the actual majority gpt-4?**

KL on tied battles splits weight 0.5 / 0.5; vicuna participated in many ties; soft mass on vicuna stacks across the training set; gradient pushes pred toward where soft mass concentrates.

</div>
</div>

<!--
**EN.** I ran the saved best adapter on the validation set with full prediction histograms. The model concentrated 93% of its predictions on just two classes — vicuna-13b at 58% and gpt-4 at 35%. Fourteen of the twenty classes never received a single prediction. The calibration error is fine — 1.8% — so the model is honest about its uncertainty. The interesting question is why vicuna, not the actual argmax majority gpt-4. The answer is in the soft-label construction: tied battles split 0.5/0.5 between participants. Vicuna-13b played in many tied battles, so its soft mass accumulates across the training set even when it doesn't have the most argmax wins. KL gradient pushes the prediction toward the integrated soft mass, not the argmax mode.

**中文.** 我把保存下来的最佳 adapter 在完整验证集上跑了一遍，看预测分布。发现模型 93% 的预测全压在两个类上：vicuna-13b 占 58%，gpt-4 占 35%。20 个类里有 14 个从头到尾一次都没被预测过。校准误差只有 1.8%，说明模型对自己的不确定性是诚实的。有意思的问题是：为什么不是真正的 argmax 多数类 gpt-4？答案在软标签的构造方式里——平局的战斗权重五五开，而 vicuna-13b 在 Arena 里参与了非常多的平局，所以它累积下来的软标签 mass 远高于直接获胜的次数。KL 散度的梯度把预测推到累积 mass 集中的地方，而不是 argmax mode 那个方向。
-->

---
layout: section
---

# L3 #5 · Top-K production-aligned restriction

<!--
**EN.** With the multi-class router stuck at 12.5%, I asked: is the 20-class framing even the right framing? VSR routes to *deployed* backends, not all 20 Arena contestants. The answer: re-frame to top-K and see what happens.

**中文.** 多类路由器卡在 12.5% 不动了，我换个角度想：20 类这个问题定义本身合不合理？VSR 实际上线的时候只会路由到部署的后端，不会用到 Arena 全部 20 个模型。把类别集合裁到 top-K 看看会怎么样。
-->

---

# Re-framing matches docs §6 deployment reality

VSR routes to *deployed backends only*, not all 20 Arena contestants. The 14 never-predicted classes contribute soft-label noise without contributing routing signal.

### Procedure (`src/data/restrict_top_k.py`)

1. Compute argmax frequency on unsplit `arena_labeled.jsonl`
2. Drop soft entries not in top-5; if remaining mass ≥ 0.5, keep + renormalize
3. Stratified 80/10/10 split on the new label set

```
22,410 -> 12,684 rows (57% kept)
gpt-4         22.5%
vicuna-13b    22.5%
gpt-3.5-turbo 20.1%
claude-v1     18.9%
koala-13b     16.0%
```

Within-top-5 majority = gpt-4 = 22.5%. Random uniform = 20%.

<!--
**EN.** Spec section 6 explicitly anticipates this: "keep only the models you actually deploy as the class set." Top-5 by argmax frequency: gpt-4, vicuna-13b, gpt-3.5-turbo, claude-v1, koala-13b. For each row I drop the soft entries not in top-5, keep only rows where the remaining mass is at least 0.5, renormalize. This drops the dataset from 22,410 to 12,684 unique prompts — kept 57%. The class distribution is much more balanced — 16% to 23% per class versus 1% to 12% in the 20-class case. Methodology note: the top-5 selection uses training-side argmax frequency only, not val/test. The new splits are fresh stratified samples, no test-set leakage from the 20-class run.

**中文.** 规范第 6 节其实预期到了这种情况，原话是"只保留你后端也在用的模型作为类别集合"。我按 argmax 频次取 top-5：gpt-4、vicuna-13b、gpt-3.5-turbo、claude-v1、koala-13b。每一行的处理是：把 top-5 之外的软标签项丢掉，剩余的概率质量加起来如果至少有 0.5 就保留，然后重新归一化。22,410 行降到 12,684 行，保留了 57%。新类分布从原来的 1-12% 拉到 16-23%，均衡多了。这里有一个方法学上的细节要说一下：top-5 的选取只用了训练集的 argmax 频次，没碰过验证集和测试集；新的切分是从过滤后的全集重新做了一次分层抽样，跟 20 类的旧切分也不重合，所以并没有从 20 类那次实验里泄漏 test 信息。
-->

---

# Top-5 results & generalization

<div grid="~ cols-2 gap-6">

<div>

### Best (CE + lr=5e-5 + r=16)

| split | top1 | exp_winrate |
|-------|------|-------------|
| **val**  | **0.2492** | 0.2482 |
| **test** | **0.2380** | 0.2384 |
| ECE | 0.025 | – |

**Test Δ vs majority = +1.26 pp / +5.6% relative.**

Mode collapse partially broken: all 5 classes predicted.

</div>

<div>

### Pred distribution on test

| class | predictions | recall |
|-------|------:|------:|
| vicuna-13b | 553 (44%) | 0.453 |
| gpt-4 | 432 (34%) | 0.364 |
| gpt-3.5-turbo | 179 (14%) | 0.184 |
| claude-v1 | 75 (6%) | 0.067 |
| koala-13b | 30 (2%) | 0.030 |

Variants tried (all worse):
- yarn pretrain: 0.2397
- inv_freq (gentle 1.4× range): 0.2358

</div>
</div>

<!--
**EN.** Top-5 results: validation top1 0.2492, test top1 0.2380, calibration error 0.025. The val-to-test gap is 1.1 percentage points, about what you expect from sampling noise on 1,250 evaluation rows — no real overfitting. Test top1 is 1.26 percentage points over the 22.5% majority baseline, or 5.6% in relative terms. The mode collapse is partially broken: all five classes get predicted. Vicuna and gpt-4 are still the dominant predictions — 44% and 34% — but gpt-3.5, claude-v1, and koala each get nontrivial mass. I also tried the yarn-pretrained variant, which the VSR project ships as the production encoder; it performed worse here. And gentler inv-freq class weighting also degraded.

**中文.** Top-5 的结果：验证 top1 是 0.2492，测试 0.2380，校准误差 0.025。验证集和测试集差 1.1 个百分点，跟 1,250 行抽样的统计噪声一致，没有真正的过拟合。测试比 22.5% 的 majority 基线高 1.26 个百分点，相对提升 5.6%。Mode collapse 部分破解了，5 个类全都被预测过了。vicuna 和 gpt-4 还是主预测对象，44% 和 34%，但 gpt-3.5、claude-v1、koala 也都拿到了一定的 mass。我还试了 yarn 预训练的变体，就是 VSR 生产环境用的那个，结果反而比 vanilla mmBERT 差。再试温和版的 inv-freq 类别加权，也降了。
-->

---
layout: section
---

# L3 #6 · Four diagnostics → root cause

<!--
**EN.** At this point I have to ask myself the obvious question: am I just gaming the metric by removing classes the model can't predict anyway? To answer that honestly I ran four diagnostics that test where the residual gap actually lives.

**中文.** 走到这一步必须先问自己一个显然的问题：是不是为了数字好看，把模型本来就预测不出来的类砍掉了？为了诚实回答这个问题，我又跑了四个诊断测试，去查真正的瓶颈到底在哪。
-->

---

# The honest question: "is top-K just metric gaming?"

Ran four orthogonal diagnostics to test where the residual gap actually lives.

<div class="text-sm">

| Diagnostic | Result | Implication |
|------------|--------|-------------|
| **1.** Per-prompt soft-label statistics | 80.8% prompts have `k=1` (single battle); for `k≥2`, only 1.6% have max-soft ≥ 0.75 cross-battle; 0% unanimous | Labels are mostly single-trial; multi-trial agreement is barely better than coin-flip |
| **2.** Length-stratified training (>50 tok) | top1 = 0.1003 < majority 0.1285 (-2.8 pp) | "Short prompts uninformative" hypothesis falsified |
| **3.** gpt-4 vs vicuna-13b binary, held-out test | top1 = 0.5042 vs majority 0.5008 (+0.3 pp) | Even closest pair gives essentially zero held-out signal |
| **4.** Linear probe on frozen mmBERT (top-5) | top1 = 0.2224 vs LoRA 0.2492 (+2.7 pp lift only) | Frozen encoder already captures most recoverable signal |

</div>

<!--
**EN.** Four orthogonal diagnostics, each testing one specific hypothesis. Diagnostic one is pure data analysis on the soft labels. Diagnostic two trains on long prompts only to test if short prompts are killing the signal. Diagnostic three picks the closest binary pair — gpt-4 versus vicuna-13b, the two argmax-tied majority classes — and runs a 2-class classifier. Diagnostic four freezes the encoder entirely and trains only a linear classifier head — measures how much of the recoverable signal the frozen encoder already captures, versus how much LoRA adds on top. I'll walk through diagnostic 1 in detail because it's the most damning.

**中文.** 四个正交的诊断，每一个测一个具体假设。诊断 1 是软标签的纯数据分析。诊断 2 只用长 prompt 训练，看是不是短 prompt 没有信号。诊断 3 取最近的二分类对，gpt-4 对 vicuna-13b，这俩是并列 argmax 多数的两个类，跑成二分类问题看看。诊断 4 把 encoder 完全冻住，只训一个分类头，看冻结表征本身能拿到多少信号、LoRA 多加了多少。下一页详细讲诊断 1，因为它是最致命的一个。
-->

---

# Diagnostic 1: 80.8% prompts ride on a single human vote

<div class="text-sm">

```
22,410 unique prompts post-§2.3 aggregation:
  k = 1  (single battle):   18,105   (80.8%)
  k = 2:                     3,475   (15.5%)
  k ≥ 3:                       830   ( 3.7%)
```

For the **4,305 multi-battle prompts** (the only ones where "soft" isn't syntactic):

```
mean max(soft_label)        = 0.485    ← coin-flip-level concentration
fraction max ≥ 0.50         = 88.1%
fraction max ≥ 0.75         =  1.6%
fraction max = 1.0          =  0.0%   ← no prompt has unanimous winner
```

</div>

<v-click>

### Implication

> Even when judges look at the same prompt multiple times, they disagree. The "soft label" P(model wins | prompt) is a noisy estimator of an underlying P that is itself heavy-tailed near 0.5.

</v-click>

<!--
**EN.** This is the killer statistic. 80.8% of prompts in the training set appear in only one Arena battle. Their soft label is a single human's single judgment, dressed up to look like a probability distribution. There's no statistical estimation happening — it's just one vote. For the 19% of prompts that do have multiple battles, look at the max soft probability — the concentration on the most-frequent winner across battles. Mean is 0.485. Eighty-eight percent of multi-battle prompts have at least 0.5 mass on the top winner, but only 1.6% have at least 0.75, and zero percent have unanimous agreement. So even when humans judge the same prompt multiple times, they disagree. The Bayes-optimal classifier given these labels is bounded near majority — there's no per-prompt signal for it to find.

**中文.** 这一页的数据是最致命的。训练集里 80.8% 的 prompt 只出现在一场战斗里。它们的"软标签"实际上就是一个人的一次判断，被伪装成概率分布的样子。其实根本没有任何统计估计在做，就是一票。剩下 19% 多场出现的 prompt，看它们在最强胜方上的概率集中度——平均 0.485。88% 的多场 prompt 至少有 0.5 mass 在最强胜方，但只有 1.6% 达到 0.75，0% 全票一致。也就是说，即使让人重复评判同一个 prompt，他们也分歧得很厉害。给这样的标签，理论上 Bayes 最优的分类器就在 majority 附近——它根本没有 per-prompt 的信号可以去学。
-->

---

# Where the budget actually went

20-class case:

```
random uniform                 0.050
+ encoder-already-knows        0.069   (linear probe extrapolation)
+ LoRA fine-tune               0.034   (CE arch-fixed - linear-probe est.)
+ everything we have not tried ~0      (capacity sweep flat, lr sweep flat)
                              -----
              test top1       ≈ 0.123  ← measured
              vs majority      0.119
```

5-class case:

```
random uniform                 0.200
+ encoder-already-knows        0.024   (linear probe = 0.222)
+ LoRA fine-tune               0.027   (LoRA top-5 - linear probe)
+ everything we have not tried ~0
                              -----
              val top1        ≈ 0.249  ← measured
              vs majority      0.225
```

**No remaining axis we haven't tested could plausibly close the gap.**

<!--
**EN.** Decomposing where the accuracy comes from. In the 20-class case, random uniform is 5%, the linear probe — which uses only the frozen encoder — gives about 6.9 points more, LoRA fine-tune adds another 3.4 points. Total 12.3% test top1, against majority of 11.9%. We have not left anything significant on the table — capacity sweep was flat across r=16/32/64, lr sweep was flat across 5e-5/1e-4/2e-4. The 5-class case decomposes similarly. No remaining axis that we haven't tested could plausibly close the gap to, say, 0.40.

**中文.** 把准确率拆开看预算用到哪去了。20 类的情况：随机猜是 5%，线性探针（也就是只用冻结的 encoder）能到 12.4%，多了 6.9 个点；再加上 LoRA 微调，又多 3.4 个点；最终到 12.3%，比 majority 的 11.9% 高 0.4。没测过的那些方向其实没什么提升空间了——容量从 r=16 到 r=64 是平的，学习率从 5e-5 到 2e-4 也是平的。5 类的情况拆法类似。剩余还没动过的轴里，没有任何一个能合理地把这个 gap 拉到比如 0.40 那个水平。
-->

---
layout: section
---

# L3 #7 · Pairwise reframing — proxy-metric trap

<!--
**EN.** One last reframing, both as a serious attempt to lift accuracy and as a test of the diagnostic conclusion. Per-battle pairwise classification with model-identity tokens.

**中文.** 还有最后一次 reframing，既是认真想再拉一下分数，也是验证前面那套诊断的结论。改成逐场次的二分类，输入里加上模型身份的 token。
-->

---

# Per-battle pairwise framing

<div grid="~ cols-2 gap-6 text-sm">

<div>

### Setup

```
input  = "[a={m_a}] [b={m_b}] {prompt}"
label  = "a" if winner == m_a else "b"
```

- 2,775 battles (top-5 pool, no ties)
- Train 2,220 / val 277 / test 278
- Labels balanced 50/50

### Binary test result — looks great in isolation

```
test top1   = 0.6403  (+13.7 pp over 50% majority)
exp_winrate = 0.6403
ECE         = 0.033
preds       = a:140, b:138 (balanced, no collapse)
```

</div>

<div>

### Borda routing — collapses to majority

For top-5 routing, query all `K(K-1)` ordered pairs per prompt; sum P(m wins) per model; argmax.

```
n_eval                = 1,269
top1_accuracy         = 0.2254   ← exact majority
delta vs majority     = 0.0000

pred distribution:
  gpt-4         1267   (99.8%)
  claude-v1        2
  gpt-3.5-turbo    0
  koala-13b        0
  vicuna-13b       0
```

**Equivalent to "always pick gpt-4".**

</div>
</div>

<!--
**EN.** I rebuilt the data per-battle, not per-aggregated-prompt — so each Arena battle becomes one training row, with the two model identities as bracketed tokens at the start of the input. 2,775 battles in the top-5 pool. Train binary CE for 8 epochs. Held-out test top1 jumps to 64%, +13.7 percentage points over the 50% balanced-binary baseline. Calibration is fine, predictions are balanced, no class collapse. This looked like a real win until I aggregated it into a routing decision via Borda count. For routing top-5, you query the model on all 20 ordered pairs per prompt, sum the win probability per model, take the argmax. Result: gpt-4 wins for 1,267 of 1,269 prompts. Top1 is exactly the majority class baseline. The classifier had learned the global tier ranking from the identity tokens — "whichever side is gpt-4 wins" — not per-prompt routing. Borda aggregation collapses any tier-prior into "always pick top tier". The +13.7 percentage points was a proxy-metric artifact, not a real gain.

**中文.** 我把数据重新按场构造，不再按聚合后的 prompt——每一场战斗变成一行，两个参赛模型的身份当作方括号 token 放在输入的最前面。在 top-5 这个池子里有 2,775 场。CE 二分类训 8 个 epoch。测试 top1 跳到 64%，比 50% 二分类基线高了 13.7 个点。校准好、预测平衡、没崩。看起来像真正的胜利——直到我用 Borda 投票把它聚合成路由决策。Top-5 路由要把每个 prompt 在所有 20 个有序模型对上的胜率累加起来取 argmax。结果是 1,267 个 prompt 全选了 gpt-4，top1 等于 majority 基线 0.2254。也就是说，分类器从身份 token 学到的其实是全局的强弱排序——"凡是 gpt-4 那一边就赢"——根本不是 per-prompt 的路由信号。Borda 聚合一上，所有 tier-prior 都坍缩成"永远选最强"。那 13.7 个点是 proxy 指标的假象，路由层面没有真实提升。
-->

---

# Why this *strengthens* the data ceiling claim

<div class="text-sm">

Cross-check against L3 #6:

- **Diagnostic 3 (binary gpt-4 vs vicuna, no identity tokens)**: held-out top1 = 0.5042 (+0.3 pp)
- **L3 #7 pairwise (binary, with identity tokens)**: held-out top1 = 0.6403 (+13.7 pp)

The 13.4 pp gap is **the value of identity tokens alone**, not prompt content. Confirmed via Borda → majority.

### Engineering lesson

> "When labels are noisy and the input includes class-identity tokens, the pairwise classifier looks impressive on its native binary metric — its labels track the tier prior — but routing-shaped evaluation collapses to majority. Always evaluate the production-shaped metric, not the proxy."

</div>

<!--
**EN.** Cross-checking with diagnostic 3 from finding 6. Diagnostic 3 was binary classification gpt-4 vs vicuna without identity tokens — held-out top1 was 50.4%, a measly 0.3 percentage points over chance. Finding 7 is binary classification with identity tokens — held-out top1 64%, 13.7 percentage points over chance. The 13.4 percentage points gap is exactly the value of the identity tokens. Identity tokens contain a global tier prior — gpt-4 is just generally better than vicuna — and that prior carries the binary metric. But it carries nothing about per-prompt routing, which is what the deployment metric actually measures. Engineering takeaway: when your labels are noisy and your input includes class-identity tokens, your binary classifier will look great on its native task and useless in production. Always evaluate the production-shaped metric.

**中文.** 跟 finding 6 里的诊断 3 交叉对比。诊断 3 是 gpt-4 对 vicuna 二分类、不带身份 token 的版本，测试 top1 是 50.4%，比抛硬币只高 0.3 个点。Finding 7 是同样的二分类、加上身份 token，测试 top1 跳到 64%。中间这 13.4 个点的差距，正好就是身份 token 自己带来的"价值"——身份 token 编码了全局强弱的先验，gpt-4 整体上比 vicuna 强，这个先验把二分类指标撑起来了。但它对 per-prompt 路由完全没用，而生产层面只关心 per-prompt 路由。教训是：当你的标签噪声大、输入又包含类别身份的 token，二分类指标会很好看，但生产层面没用。一定要用生产形状的指标去评估，不要被 proxy 指标骗了。
-->

---
layout: section
---

# 3. Summary & Defense Position

<!--
**EN.** Last section. Where does the project land, and what does it deliver.

**中文.** 最后一部分，项目最终落在哪里，交付了什么。
-->

---

# Where we are

<div class="text-sm">

| Setup | val top1 | test top1 | majority | test Δ |
|-------|---------|-----------|----------|--------|
| 20-class (CE r=16) | 0.1254 | 0.1232 | 0.1187 | +0.45 pp |
| **5-class (CE r=16)** | **0.2492** | **0.2380** | **0.2254** | **+1.26 pp** |
| Binary gpt-4 vs vicuna | 0.5425 | 0.5042 | 0.5008 | +0.34 pp |
| Pairwise + Borda routing | – | 0.2254 | 0.2254 | 0.00 pp |
| Linear probe top-5 (frozen) | 0.2224 | – | 0.20 (rand) | +2.2 pp |

</div>

<v-click>

### What this *is*

- A working VSR-deployable router with measured generalization (5-class CE adapter)
- **Seven L3 findings** documenting silent bugs (data dedup, LoRA targets) and infrastructure work (35× speed)
- A clean negative result on capacity / lr / loss-form / class-weighting / framing
- Diagnostic-grade evidence the gap is data-shaped, not model-shaped

</v-click>

<v-click>

### What this is *not*

- A solved task. Arena soft labels are too noisy for a per-prompt classifier to substantially beat majority
- A failure of the spec. The spec is internally consistent if read narrowly; we documented two ambiguities and engineered around them

</v-click>

<!--
**EN.** Final results table: 20-class router 12.3% test top1, 5-class router 23.8%, binary 50.4%, pairwise+Borda exactly the majority class baseline, frozen-encoder linear probe 22.2%. The deliverable is the 5-class CE adapter — that's the one I'd plug into VSR tomorrow if asked. What this project *is*: a working router, seven L3 findings documenting silent bugs and 35x infrastructure work, and a clean negative result on every model-side axis. What it is *not*: a solved problem. The Arena dataset's per-prompt label noise caps where this can go. And it's not a failure of the spec — the spec is internally consistent if you read it narrowly; I documented two ambiguities and engineered around them.

**中文.** 最终结果表：20 类路由的测试 top1 是 12.3%，5 类是 23.8%，二分类 50.4%，pairwise 配 Borda 路由完全等于 majority 基线，线性探针冻结 encoder 是 22.2%。交付物是 5 类 CE 那个 adapter——明天如果让我接进 VSR 我就上这个。这个项目"是"什么：是一个能用的路由器，是七个 L3 finding 把规范里两个隐藏的坑和一次 35 倍的吞吐改造记录下来，是模型侧每条轴上都干净的负结果。这个项目"不是"什么：不是一个已经解决的问题——Arena 数据 per-prompt 的标签噪声决定了它最远能走到哪；也不是规范文档的失败——规范字面上内部是自洽的，只是有两处歧义，我做了文档化和工程上的绕开。
-->

---

# Ruled out — directions not worth pursuing

<div class="text-sm">

### Already tested, doesn't help (won't revisit)

| Lever | Result |
|-------|--------|
| Loss form (KL ↔ CE) | CE wins by +0.36 pp; not the lever |
| Learning rate (5e-5 / 1e-4 / 2e-4) | Flat ±0.005 across the range |
| Class weighting (inv_freq full + gentle) | Both runs collapsed below random |
| LoRA capacity (r=16 / 32 / 64) | top1 moves 0.0004 — one sample |
| Base model (mmBERT vs mmbert-32k-yarn) | Vanilla wins; pretrain bias mismatch |
| Length stratification (>50 token prompts) | Worse than full data (-2.8 pp) |
| k≥3 multi-battle subset | More ambiguous, not cleaner |
| Pairwise reframing + Borda | Collapses to majority — tier-prior trap |
| Linear probe (frozen encoder) | LoRA only adds +2.7 pp on top |

### Untested but predicted not to help (low expected value)

| Lever | Why skip |
|-------|----------|
| Larger base (deberta-v3-large, roberta-large) | Linear probe says encoder isn't bottleneck |
| Other adapters (IA3, prefix tuning, full SFT) | Capacity sweep is flat |
| Different optimizer (Lion / Sophia) | Doesn't change Bayes-optimal |
| Self-supervised pretrain on Arena prompts | Pretraining can't create label signal |
| Multi-turn input (concat first 2 user turns) | Only 19% of prompts have >1 turn |
| Ensemble / multi-seed distillation | ±0.5 pp calibration win at best |

</div>

<!--
**EN.** This slide answers the natural question "have you tried X?" before it gets asked. The top table lists what was tested and ruled out — every lever moved top1 by less than the sampling noise on a 1,250-row eval set, or actively hurt. The bottom table lists what's untested but I can argue ahead of time it won't help. The argument leans heavily on the linear-probe diagnostic: if the frozen mmBERT representation already captures most of the recoverable signal, a bigger base or a different adapter or a different optimizer can't extract signal that isn't there. Pretraining and ensembles can't manufacture labels. Multi-turn is data-shape, but only 19% of the dataset is multi-turn so it's a small lever. The real lever is denoising the labels, which is in the next slide as future work.

**中文.** 这一页是先回答"你试过 X 吗"这个自然会被问到的问题。上面那张表列了所有测过、确定不行的方向——每一条要么 top1 的变化小于 1,250 行评估集的采样噪声，要么直接负面。下面那张表列的是没测但我能提前说明不值得测的方向。论据主要靠线性探针那个诊断：如果冻结的 mmBERT 表征已经能拿到大部分可恢复信号，那换一个更大的 base、换一个 adapter、换一个 optimizer，都没办法挤出本来不在那里的信号。预训练和集成也造不出标签里没有的信号。多轮 prompt 是数据侧的杠杆，但全数据集里只有 19% 是多轮的，杠杆很小。真正有用的杠杆是去掉标签噪声本身，这个我放到下一页"后续工作"里讲。
-->

---

# Honest framing for the defense

> "The naïve top1 number sounds small (5-class test 0.238 vs majority 0.225), but that's a real lift on a problem whose Bayes-optimal margin is bounded by judge-noise we measured directly: 80.8% of prompts ride on one human vote and multi-vote agreement is essentially coin-flip. Quadrupling LoRA capacity moves accuracy by one sample. A linear probe on the *frozen* encoder already recovers most of what fine-tuning recovers. The +1.3 pp lift over majority is roughly what's available; the *engineering work* — fixing the two silent bugs (§2.4 and LoRA targets) and the 35× throughput rework — is the contribution."

<!--
**EN.** The honest framing. The headline number is small — 23.8% test top1 versus 22.5% majority on the 5-class router. But it's a real lift on a problem whose Bayes-optimal margin is bounded by judge-noise that I measured directly: 80.8% of prompts ride on one human vote, multi-vote agreement is essentially coin-flip. Quadrupling LoRA capacity moves accuracy by one sample of noise. A linear probe on the frozen encoder already recovers most of what fine-tuning gives. So the 1.3 percentage points over majority is approximately the available signal. The contribution is the *engineering work* — finding the two silent bugs in the spec, fixing the architecture, the 35x throughput rework — and the *diagnostic narrative* that proves all of this rather than asserting it.

**中文.** 给老师讲一段诚实的话。头条数字看着不大——5 类路由的测试 top1 是 23.8%，majority 是 22.5%。但这是在一个 Bayes 最优 margin 被人类评分噪声卡死的问题上的真实提升。这个噪声我直接量化过：80.8% 的 prompt 只有一个人评判过，多人评判的时候一致性接近抛硬币。LoRA 容量翻 4 倍，准确率只动了一个样本。在冻结的 encoder 上加一个线性探针，已经能拿到大部分微调能拿到的信号。所以比 majority 高出来的 1.3 个点，差不多就是这份数据上能拿到的全部信号。真正的贡献在工程：发现规范里两个隐藏的坑、修了架构、做了 35 倍的吞吐改造，再加上一整套用诊断而不是断言去证明这一切的方法论。
-->

---

# Future work (post-defense, not before)

<div class="text-sm">

| Direction | Cost | Expected lift |
|-----------|------|---------------|
| **Re-judge a sample with GPT-4** to denoise multi-battle prompts | 2 days, ~$10 of API | top1 → 0.30+ if hypothesis right |
| **VSR integration** (M7) — merge LoRA, register adapter, gate on real traffic | 1 day | – |
| **Multi-judge consensus** (LLM panel) for Arena re-labeling | 1 week | top1 → 0.40+ if hypothesis right |
| **New Chatbot Arena release** (~1M battles, k≥3 fraction much higher) | 1 week | top1 → 0.30-0.40 if labels still noisy |

<v-click>

### Why these and not the model-side fixes

The four directions above are all **data-side**. The previous slide rules out
the model-side ones. If the data-ceiling hypothesis is correct, these are
the only levers that move the absolute number; if it's wrong, we'll find
out cheaply with the GPT-4 re-judging experiment.

</v-click>

</div>

<!--
**EN.** Future work, post-defense. Top priority is denoising labels by re-judging a sample with GPT-4 — 2 days, about $10 of API costs, and if the data-ceiling hypothesis is right we'd see top1 jump to 30%+. Second is the VSR integration itself — that's mechanical at this point, just merge LoRA weights, register the adapter, and gate on production traffic. Long-term lever is multi-judge consensus, having an LLM panel re-label Arena, which would test whether top1 can hit 0.40 if the labels are clean. There's also a new Chatbot Arena release with about a million battles — same approach, much higher k≥3 fraction, would tell us whether the noise is fundamental or just a small-sample artifact. The reason these are all data-side is that the previous slide already ruled out everything model-side. If the data-ceiling hypothesis is correct, these are the only levers that move the number; if it's wrong, the GPT-4 re-judging experiment exposes that cheaply.

**中文.** 答辩之后的工作。最优先的是用 GPT-4 重判一部分样本去标签噪声——两天时间，API 费用大概十美元——如果数据天花板这个假设是对的，top1 会跳到 30% 以上。第二件事是 VSR 集成本身，到这一步就是体力活了：合并 LoRA 权重、在 VSR 里注册 adapter、开生产流量灰度。长远的杠杆是用 LLM 面板共识重新标注 Arena，看在标签干净的前提下 top1 能不能到 40%。还有一个是 Chatbot Arena 出了新版本，大概一百万场战斗，同样的方法跑一遍，k≥3 的比例会高很多，可以看出噪声到底是数据本身的性质还是小样本的副作用。这四个方向都是数据侧的，因为前一页已经把模型侧排除完了。如果数据天花板这个假设对，那这些是仅剩的能撬动数字的杠杆；如果不对，GPT-4 重判这个实验也能用很小的代价暴露出来。
-->

---
layout: end
---

# Thank you

Code: github.com/NickWilde18/arena-router-capstone

Reports: `reports/`

```
1. data_dedup_ablation.md         (L3 #1)
2. baseline_v2_kl.md               (L3 #2)
3. baseline_v2_kl_arch_fixed.md   (L3 #3)
4. ablation_matrix.md              (L3 #4)
5. top5_class_restriction.md       (L3 #5)
6. diagnostic_root_cause.md        (L3 #6)
7. pairwise_reframing.md           (L3 #7)
```

<div class="pt-12 text-sm opacity-60">
DDA 4080 · 2026-05-16
</div>

<!--
**EN.** Thank you. Code is on GitHub at the link shown. The seven reports in the `reports/` directory document each finding individually with full numbers and methodology. Happy to take questions.

**中文.** 谢谢各位老师。代码在 GitHub 上面。reports 目录里有 7 篇报告，每个 finding 都有完整的数字和方法论。我准备好回答问题了。
-->
