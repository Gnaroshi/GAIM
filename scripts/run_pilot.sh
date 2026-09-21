#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -u -m gaim.run --config configs/pilot.json "$@"
