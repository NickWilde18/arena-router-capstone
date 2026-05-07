"""LoRA fine-tune mmBERT for prompt → best-model classification (docs §4).

Usage:
    uv run python -m src.train.ft_model_routing_lora --config configs/train.yaml
    uv run python -m src.train.ft_model_routing_lora --config configs/train.yaml --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def train(cfg: dict, smoke: bool = False) -> Path:
    raise NotImplementedError(
        "Implement on 5/9 (M4). Plan:\n"
        "  - Load tokenizer + base model (jhu-clsp/mmBERT-base) for SequenceClassification\n"
        "  - Wrap with PEFT LoraConfig (r=16, alpha=32, target=q/k/v/dense)\n"
        "  - Custom Trainer with KL-div loss against soft_label\n"
        "  - Eval callback: top-1 accuracy, expected win-rate, ECE\n"
        "  - --smoke: subsample 1k rows, 100 steps, for MPS sanity check"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--smoke", action="store_true", help="1k samples / 100 steps for sanity")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
