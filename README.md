# 의료 QA 교란 생성·평가·학습 실험

실행 위치: `저장소 루트`.
GPU는 **물리 번호 4, 5, 6, 7**만 사용합니다. 실행 프로그램 안의 `cuda:0..3`은 이 네 장을 다시 번호 붙인 것입니다.

먼저 [실험 설명서](docs/EXPERIMENT_GUIDE_KO.md)를 읽으면 질문, 데이터, 피드백, 평가와 학습의 연결을 예시로 이해할 수 있습니다. 실제로 실행한 결과와 한계는 [시험 실행 기록](docs/SMOKE_RESULTS_KO.md)에 별도로 기록합니다. [실제 사례를 따라 읽는 문서](docs/ACTUAL_EXAMPLE_KO.md)에서는 원문·생성 문장·모델 응답을 하나씩 설명합니다.

**2026-09-21: 고정 test 100문항과 기존 네 학습 모델의 평가를 완료했습니다.** 최신 내용은 [test 전체 결과](docs/TEST_RESULTS_KO.md), [실제 test 예시 6가지](docs/TEST_EXAMPLES_KO.md), [입력 자료 점검](docs/TEST_DATA_NOTES_KO.md)에 있습니다. 원문 정확도는 학습 전 66%, 학습 후 64~66%였습니다. 독립형과 적응형은 같은 한 문항에서만 잠정 오답 전환을 찾았습니다. 교란 입력 정확도는 네 학습 조건 모두 학습 전보다 낮았습니다.

## 준비된 구성

| 항목 | 설정 |
|---|---|
| 답변 모델 | Qwen/Qwen3-4B-Instruct-2507 |
| 교란 생성 모델 | 같은 체크포인트, 역할별로 별도의 새 대화 |
| 모델 버전 | `cdbee75f17c01a7cc42f958dc650907174af0554`로 고정 |
| 데이터 | MedQA 영어 4지선다, GBaker 공개 미러의 고정 revision |
| 데이터 분리 | 학습 100문항 / 자체 개발 20문항 / 원천 test에서 고정한 100문항 |
| 비교 | 원문 반복 / 독립 교란 생성 / 피드백을 본 적응형 생성, 각 5회 |
| 보조 검사 | Porsche, League of Legends, mango 고정 문장 각 1회 |
| 학습 | 네 가지 데이터 구성으로 QLoRA adapter를 각각 학습 |
| 비용 | 로컬 GPU 사용. 유료 API 호출 없음 |

임상적으로 정답이 유지되는지는 사람이 검토해야 합니다. `--allow-provisional`은 **미검토 교란을 포함하는 탐색 실험** 옵션입니다. 실행 자체는 가능하지만 결과는 잠정치로 해석합니다.

## 접속과 실행

```bash
cd GAIM
```

환경과 모델, 데이터는 프로젝트 안에 준비되어 있습니다. 시스템 Python을 바꾸지 않고 `.venv/bin/python`을 사용합니다.

평가부터 학습 후 비교까지 작은 실험을 한 번에 실행하려면 다음 명령 하나로 가능합니다. 개발 4문항·학습 12문항을 이용한 20단계 시험이며 미검토 교란을 포함합니다. 최종 test는 사용하지 않습니다.

```bash
bash scripts/run_smoke.sh my_first_run
```

진행 상태는 다른 접속 창에서 확인할 수 있습니다.

```bash
.venv/bin/python -m gaim.status my_first_run
```

아래는 각 단계를 따로 실행하는 방법입니다. 새로운 서버에 다시 설치할 때는 `bash scripts/setup.sh`를 사용합니다. 이 설치 스크립트는 원래 실험의 Ubuntu 22.04/Python 3.10 환경을 기준으로 작성했습니다.

### 1. 작은 평가 실행

개발 문항 4개만 먼저 검사합니다. 기존 결과를 덮어쓰지 않도록 새 실행 이름을 지정합니다.

```bash
bash scripts/run_pilot.sh --run-dir runs/my_dev_smoke --limit 4
```

개발 20문항 전체는 `--limit 4`를 빼면 됩니다. 모델 응답, 생성 문장, 피드백, 원문 보존 정보가 모두 저장됩니다.

```bash
bash scripts/run_pilot.sh --run-dir runs/my_dev_pilot
cat runs/my_dev_pilot/report.md
```

### 2. 학습용 교란 생성

학습 원천에서 12문항을 골라 시험합니다. 개발·test 문항은 학습에 들어가지 않습니다.

```bash
.venv/bin/python -u -m gaim.run \
  --config configs/train_generation.json \
  --run-dir runs/my_train_smoke --limit 12
```

학습 100문항 전체는 `--limit 12`를 빼면 됩니다. 생성 결과가 100문항 모두 유효하다는 뜻은 아닙니다. 두 생성 방법에서 사용할 후보가 모두 있는 공통 문항만 네 학습 조건에 똑같이 사용합니다.

### 3. 네 가지 추가 학습

먼저 생성된 `review.csv`를 검토합니다. 기계적인 연결 확인만 하려면 다음처럼 미검토 후보 사용을 명시합니다.

```bash
.venv/bin/python -u -m gaim.training \
  --source-run runs/my_train_smoke \
  --output-dir runs/my_training_smoke \
  --steps 20 --allow-provisional
```

네 GPU에서 `clean`, `random`, `independent`, `adaptive` adapter를 하나씩 학습합니다. `random`은 세 고정 배경문장 중 seed로 고른 단순 증강 기준입니다. 대규모 무작위 LLM 생성 데이터를 의미하지 않습니다. 실제 연구용 학습은 교란을 검토한 뒤 `--allow-provisional` 없이 실행합니다.

### 4. 학습하지 않은 같은 문제로 재평가

```bash
.venv/bin/python -u -m gaim.replay \
  --source-run runs/my_dev_smoke \
  --training-dir runs/my_training_smoke \
  --output-dir runs/my_replay_smoke \
  --allow-provisional
cat runs/my_replay_smoke/report.md
```

학습 전 모델과 네 adapter에 **같은 원문, 같은 선택지, 같은 추가 문장**을 입력합니다. 이 단계는 고정된 교란에 대한 일반화를 검사합니다. 새로 학습한 모델을 다시 공격하는 실험과는 다릅니다.

### 5. 최종 평가용 100문항

개발과 설정 선택이 끝났을 때만 사용합니다. 최초 준비 과정에서는 test 문항을 평가하지 않았으며, 이후 사용자 요청으로 `runs/test_20260921`에서 고정 100문항을 평가했습니다. 기존 네 adapter의 같은 입력 비교는 `runs/test_20260921_replay`에 있습니다. 아래는 실행 방법을 보여 주는 명령입니다.

```bash
bash scripts/run_evaluation.sh --run-dir runs/final_evaluation_v1
```

학습 후 비교는 위 replay 명령의 `--source-run`을 이 최종 실행 폴더로 바꿉니다. 이 100문항은 이제 결과를 확인한 표본입니다. test 결과를 보고 설정을 계속 바꿔 같은 표본을 재평가하면 후속 탐색 평가로 표시해야 합니다.

## 결과 읽기

| 파일 | 내용 |
|---|---|
| `run.json` | 문항 ID, 모델 revision, 설정, 코드·데이터 해시 |
| `questions.jsonl` | 이번 실행의 원문, 선택지, 정답과 출처 |
| `worker_0.jsonl` 등 | 실제 모델 응답, 정답 여부, 적용 문장, 토큰 수, 시간 |
| `generations_0.jsonl` 등 | 생성기 입력, 받은 피드백, 생성 원문 |
| `report.md` / `summary.json` | 사람이 읽는 보고서 / 프로그램용 집계 |
| `review.csv` | 교란의 정답 보존을 사람이 검토하는 표 |
| `environment.txt` | 실제 설치된 패키지 버전 |
| `COMPLETE` | 실행과 집계가 정상 종료되었을 때만 생성 |

`review.csv`의 `clinical_review`에 `approved` 또는 `rejected`, `reviewer`에 검토자, `review_note`에 이유를 적습니다. `approved`는 단순히 문장이 자연스럽다는 뜻이 아니라, 해당 문제에서 추가 내용이 정답을 바꾸지 않는다고 검토했다는 뜻입니다. 모호하면 `pending`으로 남깁니다. 재집계해도 기존 검토 메모는 보존합니다.

```bash
.venv/bin/python -m gaim.metrics --run-dir runs/my_dev_smoke --k 5
```

보고서의 잠정 전환율은 자동 규칙을 통과한 교란에서 관측한 오답입니다. 승인된 오답 수는 사람이 검토한 사례만 셉니다. 미검토 사례가 남아 있으면 이를 전체 공격 성공률의 확정치로 해석하지 않습니다.

## 중단, 이어 실행, 상태 확인

SSH 창을 닫아도 계속하려면 새 터미널 세션에서 실행합니다.

```bash
tmux new -s gaim-pilot
bash scripts/run_pilot.sh --run-dir runs/my_long_pilot
```

`Ctrl-b`를 누른 뒤 `d`로 분리하고, `tmux attach -t gaim-pilot`으로 돌아옵니다. GPU 상태는 `nvidia-smi -i 4,5,6,7`로 확인합니다. 실행 중인 worker 기록은 해당 실행 폴더의 `worker_*.log`에서 확인합니다.

평가가 정상적으로 중단된 경우 같은 설정에 `--resume`을 붙이면 완료된 요청을 재사용합니다.

```bash
bash scripts/run_pilot.sh --run-dir runs/my_dev_smoke --limit 4 --resume
```

설정·코드·데이터·환경이 바뀌었거나 실행 오류가 기록되어 있으면 재개를 거부합니다. 오류 로그를 남겨 두고 새 폴더에서 실행합니다. 학습은 새 출력 폴더를 쓰며, 최적화 중간 지점 복원은 현재 구현 범위에 포함하지 않습니다. 기존 adapter를 실수로 덮어쓰지 않습니다.

## 구성 검증

```bash
.venv/bin/python -m pytest -q
```

테스트 안의 가상 문항과 응답은 소프트웨어 검사 전용입니다. 실제 의료 QA 실험 결과에 포함하지 않습니다.

## 관련 자료

- [MedQA 원 저장소](https://github.com/jind11/MedQA)
- [이번 실행에 사용하는 데이터 미러](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
- [사용 모델과 라이선스](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [PAIR](https://arxiv.org/abs/2310.08419): 이전 응답을 이용한 반복 수정 구조의 참고 문헌
- [MedFuzz](https://arxiv.org/abs/2406.06573): 의료 QA의 적대적 변형 연구
- [PEFT 양자화·QLoRA 문서](https://huggingface.co/docs/peft/developer_guides/quantization)

이 구현을 PAIR/MedFuzz의 원 실험 재현이나 새로운 최초 방법으로 주장하지 않습니다. 이번 수업에서 확인할 가설과 통제 조건을 구현한 출발점입니다.
