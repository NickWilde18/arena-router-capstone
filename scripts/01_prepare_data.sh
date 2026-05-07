#!/usr/bin/env bash
# End-to-end data preparation: download → filter → label construct → split.
# Requires HF_TOKEN env var (Arena dataset is gated).

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set. See README.md → Quickstart." >&2
    exit 1
fi

uv run python -m src.data.download
uv run python -m src.data.filter
uv run python -m src.data.label_construct
uv run python -m src.data.split

echo
echo "Data prep complete."
ls -lh data/processed/
