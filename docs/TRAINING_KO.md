# 전체 MedQA를 네 tmux pane에서 실행하기

**현재 실행:** 원본 train 10,178행 중 정답이 충돌하는 중복 4행을 제외한 10,174문제. 모델당 20,348입력 × 2 epochs. test 1,273문제 전체를 학습 후 자동 평가한다.

이전 10문제 학습과 다른 실행이다. [비교 방법과 문장 예시](FULL_EXPERIMENT_KO.md)를 먼저 읽으면 각 조건의 뜻을 알 수 있다.

## 준비

프로젝트 폴더에서 실행한다. 기존 `.venv`, 모델 캐시, `data/medqa/sources`를 사용한다. 새 서버의 설치와 자료 다운로드는 [재현 안내](REPRODUCING_KO.md)를 참고한다.

```bash
.venv/bin/python -m gaim.full_data   --experiment runs/full_medqa_20260922 --gpus 0,1,2,3
```

CPU 작업이다. 전체 원문과 평가 입력, 설정을 저장한다. GPU를 할당받은 번호가 다르면 `--gpus`와 아래 네 명령을 함께 바꾼다. 이미 시작한 실행의 GPU 매핑은 변경하지 않는다.

## pane마다 명령 하나

사용자의 최신 할당은 **GPU 0,1,2,3**이다. 프로젝트 폴더에서 다음을 실행한다.

```bash
bash scripts/full_pane.sh clean 0 runs/full_medqa_20260922
```

```bash
bash scripts/full_pane.sh random 1 runs/full_medqa_20260922
```

```bash
bash scripts/full_pane.sh flip 2 runs/full_medqa_20260922
```

```bash
bash scripts/full_pane.sh near_miss 3 runs/full_medqa_20260922
```

기존 pane에서 실행한다. pane을 종료하는 명령은 포함하지 않았다. 작업이 끝나거나 실패하면 원래 shell로 돌아온다.

## 자동으로 진행되는 순서

1. **GPU 대기:** 해당 GPU의 여유 메모리가 22,000MiB 이상인지 60초마다 확인한다. 다른 작업을 종료하지 않는다. 따라서 GPU가 비워진 뒤 다음 확인 주기에 시작한다.
2. **학습 자료 선택을 위한 채점:** 네 GPU가 원문 전체를 4분할해 공개 모델로 채점한다. 학습 문제마다 원문과 후보 4개를 채점한다. test 기준 결과도 저장한다.
3. **학습 입력 구성:** 네 분할이 끝나면 같은 원본 문제를 사용한 네 조건의 자료를 만든다.
4. **추가 학습:** 각 pane은 자신의 조건 하나를 GPU 한 장에서 학습한다.
5. **평가:** 완료된 모델은 전체 test의 원문·교란 입력 12,730개를 자동 평가한다.

GPU가 비었다고 즉시 가중치를 갱신하는 것은 아니다. 새 교란의 점수를 계산해 학습 자료를 고르는 단계가 먼저 필요하다. **위 명령 한 번으로 채점→학습→평가까지 연결되어 있다.** 에이전트가 계속 지켜보거나 다시 실행할 필요는 없다.

## 학습 설정과 메모리

같은 공개 Qwen3-4B-Instruct-2507을 BF16으로 올리고, 작은 추가 가중치인 LoRA를 학습한다. 원래 모델 전체를 처음부터 학습하지 않는다. rank 8, alpha 16, 학습률 0.0001, 2 epochs, 갱신당 입력 16개다. 마지막 12개 입력도 버리지 않아 모델당 2,544번 갱신한다.

시작 시 가장 긴 실제 입력으로 한 번에 처리할 크기를 1·2·4·8·16 중 고른다. 메모리 90% 이내에서 가능한 크기를 쓰며, 원문을 자르지 않는다. 긴 문제를 처리하기 위해 중간 계산 일부를 다시 계산하는 gradient checkpointing을 켠다. 모든 단계가 항상 90% 메모리를 쓰는 것은 아니다.

## 출력과 재개

`runs/full_medqa_20260922/` 아래에 저장된다.

| 경로 | 내용 |
|---|---|
| `experiment.json` | 전체 문제 수, 원천 버전, 생성 규칙, GPU 매핑 |
| `scored/shard_*.jsonl` | 공개 모델의 후보별 실제 답과 점수 |
| `training/selection_counts.json` | 각 정책에서 조건을 만족한 수와 무작위 대체 수 |
| `training/{조건}.jsonl` | 실제 학습 입력 20,348개 |
| `arms/{조건}/adapter/` | 추가 학습 가중치 |
| `arms/{조건}/train_state.json` | 학습 설정·완료 epoch·배치·메모리 |
| `arms/{조건}/test_summary.json` | 원문·오답 의견·정답 의견별 평가 |
| `evaluation/base_summary.json` | 추가 학습하지 않은 공개 모델의 평가 |
| `logs/{조건}.log` | 해당 pane의 출력 |

중단되면 같은 명령으로 재개한다. 채점은 완료한 문제 다음부터, 학습은 마지막으로 저장한 epoch 다음부터 진행한다. 완료한 결과는 다시 학습하지 않는다. 설정을 바꾸려면 새 실행 폴더를 사용한다.

현재 명령은 첫 전체 비교다. 여러 seed, 자유로운 LLM 문장 생성, 외부 MedDistractQA 평가는 아직 자동 실행에 포함하지 않았다.
