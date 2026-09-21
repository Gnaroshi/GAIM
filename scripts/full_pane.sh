#!/usr/bin/env bash
# 이 명령은 기존 pane을 사용한다. 완료나 실패 뒤에도 tmux pane을 종료하지 않는다.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/full_pane.sh ARM GPU EXPERIMENT_DIR" >&2
  exit 2
fi
arm="$1"
gpu="$2"
experiment="$3"
mkdir -p "$experiment/logs"
.venv/bin/python -u -m gaim.full_pipeline --experiment "$experiment" --arm "$arm" --gpu "$gpu" \
  2>&1 | tee -a "$experiment/logs/$arm.log"
