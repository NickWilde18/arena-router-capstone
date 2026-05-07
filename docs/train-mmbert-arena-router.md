# 使用 `lmsys/chatbot_arena_conversations` 训练 vLLM Semantic Router 的 mmBERT 路由分类器

本文档说明如何利用 LMSYS Chatbot Arena 的成对偏好数据集，训练一个可以直接接入 [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)(VSR) 的 mmBERT 分类头，完成 **"给定 prompt → 选择最优后端模型"** 的语义路由任务。

---

## 1. 背景与问题定义

### 1.1 VSR 的路由机制
VSR 采用 **signal-decision** 架构：每条 prompt 先经过若干个 BERT-like 分类器产生 "信号"（domain / PII / jailbreak / modality / feedback / fact-check），再由决策层挑选后端模型或推理链。v0.2 "Athena" 的生产默认编码器是 `llm-semantic-router/mmbert-32k-yarn`（基于 `jhu-clsp/mmBERT-base` 并经 YaRN 扩到 32K 上下文），使用 HuggingFace PEFT LoRA 微调。

现有分类器训练脚本集中在 `src/training/model_classifier/`，形式均为 `{text, label}` JSONL 的**单标签序列分类**。

### 1.2 Arena 数据的特性
`lmsys/chatbot_arena_conversations` 包含约 33K 条人类成对偏好记录，核心字段：

| 字段 | 含义 |
|---|---|
| `conversation_a`, `conversation_b` | 两个模型的完整对话（含用户 prompt） |
| `model_a`, `model_b` | 参与对战的模型名 |
| `winner` | `model_a` / `model_b` / `tie` / `tie (bothbad)` |
| `language`, `turn`, `openai_moderation`, `toxic_chat_tag` | 元信息 |

这是**成对偏好数据**，无法直接当作 `{text, label}` 分类样本。必须先做标签构造。

### 1.3 目标任务
**Model-quality predictor（模型质量预测器）**：输入为用户 prompt，输出为 N 个候选后端模型的胜率分布；路由时取 argmax。这与 VSR 的 intent/domain 分类头结构一致，可直接复用训练与推理管线。

---

## 2. 标签构造

### 2.1 候选模型集合
从数据集中的 `model_a` ∪ `model_b` 统计频次，保留样本量 ≥ 阈值（建议 500）的模型，得到类别集合 `M = {m_1, ..., m_N}`。典型 N 在 10~20 之间（gpt-3.5-turbo、gpt-4、claude-v1、vicuna-13b 等）。

### 2.2 从成对战斗到单样本标签
每条 battle 贡献一条或两条训练样本：

- `winner == model_a` → 样本 `(prompt, label=m_a, weight=1.0)`
- `winner == model_b` → 样本 `(prompt, label=m_b, weight=1.0)`
- `winner == tie` → 同时产出两条 `(prompt, label=m_a, 0.5)` 和 `(prompt, label=m_b, 0.5)`
- `winner == tie (bothbad)` → 丢弃（无正向信号）

`prompt` 取 `conversation_a[0]["content"]`（首轮用户消息）。多轮对话下推荐只使用首轮以稳定输入分布；若要保留多轮，用 `\n\n` 拼接截断到 32K tokens（mmBERT-32K 上限）。

### 2.3 聚合为软标签（推荐）
将相同 prompt 的多条记录聚合：
```
P(m | prompt) = Σ weight_i · 1[label_i == m] / Σ weight_i
```
得到软分布 `y ∈ Δ^N`，训练时使用 **soft cross-entropy / KL 散度**，比 hard-label 更稳健（因为 arena 的 judge 噪声很大）。

### 2.4 去重与过滤
- 按 `question_id` 去重
- 过滤 `toxic_chat_tag != safe` 的样本（避免污染）
- 过滤 `language != en` 或保留多语言（mmBERT 本身多语能力良好，可保留）
- 按小写+去空格后的 prompt hash 去近似重复

---

## 3. 数据 Schema（VSR 兼容）

在 `src/training/model_classifier/model_routing_lora/` 下新建任务目录。输出两个 JSONL：

```jsonl
{"text": "Write a Python function to ...", "label": "gpt-4", "soft_label": {"gpt-4": 0.6, "claude-v1": 0.3, "vicuna-13b": 0.1}}
{"text": "What is the capital of France?", "label": "gpt-3.5-turbo", "soft_label": {"gpt-3.5-turbo": 0.5, "gpt-4": 0.5}}
```

- `label`: argmax 类别，供 VSR 默认 hard-label 流程使用
- `soft_label`: 训练时可选使用的完整分布
- 按 80/10/10 切分为 `train.jsonl` / `val.jsonl` / `test.jsonl`

`label2id.json` 列出类别顺序。

---

## 4. 训练脚本

参考 `src/training/model_classifier/classifier_model_fine_tuning_lora/ft_linear_lora.py` 新增 `ft_model_routing_lora.py`。关键配置：

```python
BASE_MODEL = "jhu-clsp/mmBERT-base"          # 或 llm-semantic-router/mmbert-32k-yarn
NUM_LABELS = len(label2id)
MAX_LEN = 2048                                # 首轮 prompt 通常足够；上下文长任务改 32K

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["query", "key", "value", "dense"],
    task_type=TaskType.SEQ_CLS,
)

training_args = TrainingArguments(
    learning_rate=2e-4,                       # LoRA 可用较大 LR
    per_device_train_batch_size=32,
    num_train_epochs=3,
    warmup_ratio=0.06,
    weight_decay=0.01,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=500,
    metric_for_best_model="top1_winrate",
    load_best_model_at_end=True,
)
```

**Loss**：若使用 `soft_label`，自定义 `compute_loss` 为 `F.kl_div(log_softmax(logits), soft_label, reduction="batchmean")`；否则常规 `CrossEntropyLoss`。

**评估指标**：
- Top-1 accuracy（是否选到 argmax 模型）
- **Expected win-rate**：在 test 集上按预测模型查询其对应的 soft_label 概率，取均值。这是真实路由收益的上界近似。
- Calibration（ECE）

---

## 5. 与 VSR 集成

1. 训练产物：`adapter_model.safetensors` + `label2id.json`
2. 合并 LoRA 权重到 base 模型：`model.merge_and_unload()` 后 `save_pretrained("model_routing_classifier/")`
3. 在 VSR 配置中作为新信号注册：
   ```yaml
   classifiers:
     - name: model_routing
       path: ./models/model_routing_classifier
       labels: ./models/model_routing_classifier/label2id.json
       signal: best_model
   ```
4. 决策层用 `best_model` 信号直接映射到后端 vLLM endpoint。

Rust/Candle 侧沿用 `candle-binding/` 现有 mmBERT 加载路径，无需额外改动。

---

## 6. 限制与注意事项

- **类别不均衡**：gpt-3.5-turbo / gpt-4 样本远多于小模型，需按类别加权采样，或在 loss 中使用类频倒数权重。
- **模型集合漂移**：Arena 中的模型（vicuna-13b、koala 等）可能与你线上后端不一致。可以：(a) 重映射到你实际部署的模型族；(b) 只保留你后端也在用的模型作为类别集合。
- **prompt 分布偏差**：Arena 偏向英文、开放对话；如你线上流量以代码/中文为主，需补充自有数据或用 LLM-judge 在 `verify_text_classification_datasets.py` 上做标签审计。

---

## 7. 快速执行清单

1. `datasets.load_dataset("lmsys/chatbot_arena_conversations")` 下载
2. 运行数据处理脚本（§2 + §3）产出三份 JSONL
3. 拷贝并修改 `ft_linear_lora.py` → `ft_model_routing_lora.py`
4. `bash scripts/train-mmbert32k-gpu.sh model_routing`（或单任务启动）
5. 评估 top-1 与 expected win-rate
6. 合并 LoRA、注册到 VSR 配置、端到端灰度
