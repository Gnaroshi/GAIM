#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_name="${1:-smoke_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$run_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo '실행 이름에는 영문, 숫자, 밑줄, 하이픈만 사용하세요.' >&2
  exit 2
fi
mkdir -p logs
record_status() {
  run_status=$?
  if [[ "$run_status" -eq 0 && ! -f "runs/${run_name}_replay/COMPLETE" ]]; then
    run_status=130
  fi
  printf "%s\n" "$run_status" > "logs/${run_name}.exit"
}
trap record_status EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for suffix in dev train adapters replay; do
  if [[ -e "runs/${run_name}_${suffix}" ]]; then
    echo "기존 결과가 있습니다: runs/${run_name}_${suffix}. 새 실행 이름을 사용하세요." >&2
    exit 2
  fi
done
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python -u -m gaim.run --config configs/pilot.json --run-dir "runs/${run_name}_dev" --limit 4
.venv/bin/python -u -m gaim.run --config configs/train_generation.json --run-dir "runs/${run_name}_train" --limit 12
.venv/bin/python -u -m gaim.training --source-run "runs/${run_name}_train" --output-dir "runs/${run_name}_adapters" --steps 20 --allow-provisional
.venv/bin/python -u -m gaim.replay --source-run "runs/${run_name}_dev" --training-dir "runs/${run_name}_adapters" --output-dir "runs/${run_name}_replay" --allow-provisional
echo "완료. 평가: runs/${run_name}_dev/report.md"
echo "학습 후 비교: runs/${run_name}_replay/report.md"
