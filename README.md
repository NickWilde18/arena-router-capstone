# Arena Router · mmBERT-based Model Routing for vLLM Semantic Router

> **DDA 4080 Capstone Project** · CUHK-Shenzhen · 2026 Spring

A model-quality predictor that maps a user prompt to the best backend LLM,
trained on the LMSYS Chatbot Arena pairwise preference dataset and deployed
as a classifier head into [`vllm-project/semantic-router`](https://github.com/vllm-project/semantic-router) (VSR).

## TL;DR

- **Input**: user prompt (single-turn user message, ≤ 2048 tokens)
- **Output**: probability distribution over N candidate backend models
- **Architecture**: `jhu-clsp/mmBERT-base` + PEFT LoRA (r=16) seq-classification head
- **Training data**: ~25K unique prompts derived from `lmsys/chatbot_arena_conversations` (33K pairwise battles → soft labels)
- **Loss**: KL divergence against soft labels (Arena judges are noisy)
- **Integration**: HuggingFace adapter merged into VSR's classifier slot, exposed as a `best_model` signal

The full design rationale and label-construction logic are in
[`docs/train-mmbert-arena-router.md`](docs/train-mmbert-arena-router.md) — that
document is the project specification.

## Roadmap (5/7 → 5/16, 9 days)

| Date  | Milestone | Output |
|-------|-----------|--------|
| 5/7   | M1: Repo skeleton + data download                              | This repo + `data/raw/arena.parquet` |
| 5/8   | M2-M3: Filtering + soft-label construction + EDA               | `train/val/test.jsonl` + `label2id.json` |
| 5/9   | M4: Training script + Mac MPS smoke test (1k samples)          | 100-step run completes without OOM |
| 5/10  | M5: Baseline training on RTX 4070 (hard-label CE)              | `outputs/adapter_v1/` + curves |
| 5/11  | L3 ablation 1: soft-label KL + class-balanced loss             | `outputs/adapter_v2/` |
| 5/12  | L3 ablation 2: Bradley-Terry skill rerank                      | `outputs/adapter_v3/` + ablation table |
| 5/13  | M6: Top-1 / expected-winrate / ECE / confusion matrix          | `reports/eval.md` |
| 5/14  | M7: VSR end-to-end demo + upstream PR                          | Demo recording + PR link |
| 5/15  | M8: Report + slides + rehearsal                                | `reports/slides.md` |
| 5/16  | **Capstone defense (morning)**                                 |  |

## Quickstart

The Arena dataset is gated. Request access on the
[HuggingFace dataset page](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)
first, then export your token:

```bash
export HF_TOKEN=hf_xxxxxxxx
```

Then:

```bash
# Install (Python 3.11; uv-managed)
uv sync

# Data prep pipeline
uv run python -m src.data.download
uv run python -m src.data.filter
uv run python -m src.data.label_construct
uv run python -m src.data.split

# Train (MPS for smoke; CUDA for real)
uv run python -m src.train.ft_model_routing_lora --config configs/train.yaml --smoke   # Mac M-series
uv run python -m src.train.ft_model_routing_lora --config configs/train.yaml           # RTX 4070

# Evaluate + merge LoRA for VSR integration
uv run python -m src.eval.metrics --adapter outputs/adapter_v1
uv run python -m src.integration.merge_lora --adapter outputs/adapter_v1 --out outputs/merged_v1
```

## Layout

```
.
├── docs/                Project spec + references
├── src/
│   ├── data/            Download · filter · label · split
│   ├── train/           LoRA fine-tuning + soft-CE/KL loss
│   ├── eval/            Top-1, expected win-rate, ECE, error analysis
│   └── integration/     Merge LoRA, register into VSR config
├── configs/             YAML configs (data.yaml, train.yaml)
├── scripts/             End-to-end shell pipelines
├── notebooks/           EDA, error inspection
├── reports/             Ablation tables, defense materials
├── data/
│   ├── raw/             gitignored — Arena dump
│   └── processed/       gitignored — JSONL splits
└── outputs/             gitignored — checkpoints, adapters
```

## Citations

```bibtex
@misc{zheng2023judging,
  title  = {Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
  author = {Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and others},
  year   = {2023}
}
```

## License

Apache 2.0 (matches upstream vLLM Semantic Router).
