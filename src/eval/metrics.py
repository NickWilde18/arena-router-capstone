"""Routing-quality metrics (docs §4 'Evaluation metrics').

  - top1_accuracy:   prediction matches argmax of soft_label
  - expected_winrate: mean over test set of soft_label[pred_class];
                      estimated probability that the routed model wins.
                      Upper-bound approximation of real routing payoff.
  - ECE:             expected calibration error
  - confusion matrix: per-class hits/misses for error analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path


def evaluate(adapter_dir: Path, test_jsonl: Path, label2id: Path) -> dict:
    raise NotImplementedError("Implement on 5/13 (M6).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--label2id", type=Path, default=Path("data/processed/label2id.json"))
    args = parser.parse_args()
    evaluate(args.adapter, args.test, args.label2id)


if __name__ == "__main__":
    main()
