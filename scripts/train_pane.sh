#!/usr/bin/env bash
# 기존 tmux pane에서 한 조건만 실행한다. 새 tmux 창이나 다른 학습을 시작하지 않는다.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/train_pane.sh ARM GPU OUTPUT_DIR" >&2
  exit 2
fi
arm="$1"
gpu="$2"
output_dir="$3"
if [[ ! -d "$output_dir/$arm" ]]; then
  echo "먼저 --prepare-only로 학습 폴더를 준비하세요: $output_dir/$arm" >&2
  exit 2
fi
.venv/bin/python -u -m gaim.train_arm --output-dir "$output_dir" --arm "$arm" --gpu "$gpu" \
  2>&1 | tee -a "$output_dir/$arm/worker.log"
