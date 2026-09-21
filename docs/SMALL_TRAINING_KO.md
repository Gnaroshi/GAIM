# 과거 10문제 학습 — 전체 실행이 아님

현재 실행은 [전체 학습 명령](TRAINING_KO.md)을 사용한다. 아래는 이전 작은 실행의 기록이다.

# GPU 네 장에서 직접 학습하기

**각 tmux pane에서 명령 하나씩 실행합니다.** 한 pane이 모델 하나를 학습합니다. 네 pane은 서로 기다리지 않습니다. 에이전트가 학습을 시작하거나 계속 상태를 확인할 필요가 없습니다.

## 무엇을 학습하나요?

네 모델 모두 **Qwen 팀이 이미 학습해 공개한 Qwen3-4B-Instruct-2507**에서 출발합니다. 이 프로젝트가 추가로 가르치는 자료만 다릅니다.

| pane / GPU | 학습 입력 |
|---|---|
| 1 / 4 | 원래 의료 문제 10개를 각각 두 번 |
| 2 / 5 | 원문 10개 + 같은 문제에 고정 취미 문장을 붙인 10개 |
| 3 / 6 | 원문 10개 + 이전 답을 보지 않고 생성한 문장을 붙인 10개 |
| 4 / 7 | 원문 10개 + 이전 답과 정오를 보고 생성한 문장을 붙인 10개 |

모든 입력의 정답은 **원본 의료 문제의 정답**입니다. 네 모델은 각각 처음 공개 모델에서 출발하며, 앞 pane의 학습 결과를 이어받지 않습니다. 문장 생성도 이미 끝난 기록을 사용하므로 아래 명령은 새 문장을 만들지 않습니다.

## 기존보다 메모리를 어떻게 더 쓰나요?

| 항목 | 기존 작은 학습 | 이번 설정 |
|---|---|---|
| 모델 저장 형식 | 4비트 압축 | BF16, 16비트로 로드 |
| 중간 계산 | 저장량을 줄이고 나중에 다시 계산 | 저장해서 재계산을 줄임 |
| 한 번에 처리하는 입력 | 1개 | 메모리에 맞춰 최대 20개 자동 선택 |
| 한 번 갱신할 때 사용하는 입력 | 2개 | 20개 |
| 가중치 갱신 횟수 | 20회 | 20회 |
| 총 처리 입력 수 / 모델 | 40개 | 400개 |

각 프로세스가 시작할 때 실제 입력으로 짧게 메모리를 측정하고, GPU 메모리 약 90% 안에서 가장 큰 배치를 선택합니다. 메모리가 부족하면 여러 번에 나눠 계산하되 **20개 입력마다 한 번 갱신**하는 조건은 같습니다. 입력을 길게 늘리거나 의미 없는 데이터로 메모리를 채우지 않습니다. 실제 사용량이 반드시 90%가 되는 것은 아닙니다.

처음 출력에 선택한 배치와 메모리 사용량이 나오고, 학습이 끝나면 실제 최대 사용량을 저장합니다. 이 설정의 GPU 사용량과 학습 결과는 아직 측정하지 않았습니다. **기존 4비트 결과와 같은 설정을 재현하는 명령은 아닙니다.**

## 준비는 한 번만

기존 서버에는 이 단계까지 준비해 두었습니다. 다른 clone에서 실행하거나 새 학습 폴더를 만들 때만 다음 명령이 필요합니다. GPU 학습은 시작하지 않습니다.

```bash
GAIM_CUDA_DEVICES=4,5,6,7 .venv/bin/python -m gaim.train_arm \
  --source-run repro/reference/train --output-dir runs/user_training_v2 \
  --prepare-only --steps 20 --effective-batch 20 --allow-provisional
```

`--allow-provisional`은 추가 문장의 의학적 무관성을 아직 사람이 확인하지 않았다는 뜻입니다. 이 자료로 탐색 실험을 진행한다는 것을 기록합니다.

## pane마다 실행할 명령

아래는 **서버의 GAIM 프로젝트 폴더 안에서** 실행하는 명령입니다. GPU 번호는 원래 서버 기준입니다. 다른 서버에서는 할당받은 네 번호로 바꾸고 준비 단계에도 같은 번호를 지정하세요.

**pane 1 — 원문으로 학습, GPU 4**

```bash
bash scripts/train_pane.sh clean 4 runs/user_training_v2
```

**pane 2 — 고정 취미 문장을 붙인 문제로 학습, GPU 5**

```bash
bash scripts/train_pane.sh random 5 runs/user_training_v2
```

**pane 3 — 이전 답을 보지 않고 만든 문장을 붙인 문제로 학습, GPU 6**

```bash
bash scripts/train_pane.sh independent 6 runs/user_training_v2
```

**pane 4 — 이전 답을 보고 만든 문장을 붙인 문제로 학습, GPU 7**

```bash
bash scripts/train_pane.sh adaptive 7 runs/user_training_v2
```

## 끝나면 무엇이 생기나요?

각 폴더에 `adapter/`(추가 학습 가중치), `training_loss.jsonl`(학습 진행), `training_result.json`(완료 여부·배치·메모리)이 저장됩니다. 네 모델이 모두 끝나면 상위 `training.json`의 상태가 `complete`가 됩니다.

학습 완료 뒤 같은 test 입력으로 비교하려면 다음을 한 번 실행합니다. **네 pane의 학습이 모두 끝난 뒤에만** 실행하세요.

```bash
GAIM_CUDA_DEVICES=4,5,6,7 .venv/bin/python -m gaim.replay \
  --source-run repro/reference/test --training-dir runs/user_training_v2 \
  --output-dir runs/user_training_v2_test --allow-provisional
```

이때 표의 `untrained`는 **우리가 추가 학습하지 않은 Qwen 공개 모델의 저장된 응답**입니다. 다른 네 행은 이번에 학습한 모델의 새 응답입니다.

중단된 학습을 처음부터 다시 할 때는 새 폴더 이름으로 준비하고 네 명령의 폴더 이름도 바꾸세요. 기존 결과를 지우거나 자동으로 덮어쓰지 않습니다.
