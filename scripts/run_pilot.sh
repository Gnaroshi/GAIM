#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_HOME="$PWD/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -u -m gaim.run --config configs/pilot.json "$@"
