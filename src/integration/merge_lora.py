"""Merge LoRA adapter into base mmBERT and export for VSR registration (docs §5).

After training, peft's `merge_and_unload()` folds LoRA into base weights so the
output is a plain HuggingFace model directory that can be registered as a
classifier in VSR's router config.

Usage:
    uv run python -m src.integration.merge_lora \\
        --adapter outputs/adapter_v1 \\
        --out outputs/merged_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path


def merge_and_export(adapter_dir: Path, out_dir: Path) -> Path:
    raise NotImplementedError("Implement on 5/14 (M7).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    merge_and_export(args.adapter, args.out)


if __name__ == "__main__":
    main()
