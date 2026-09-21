# 다른 서버에서 실행하기

> 이 문서는 과거 작은 실험의 기록/재현 안내다. 현재 전체 MedQA 실행은 [전체 실험 설명](FULL_EXPERIMENT_KO.md)과 [전체 학습 명령](TRAINING_KO.md)을 따른다. 설치·다운로드 절차는 재사용할 수 있다.

이 프로젝트는 Qwen 팀의 공개 모델에 의료 문제를 추가 학습시켜 비교합니다. **“학습 전”은 이 프로젝트의 추가 학습 전**을 뜻합니다. 네 학습 데이터의 차이는 [첫 화면의 표](../README.md#어떤-모델을-비교하나요)를 보세요.

## 1. 설치

```bash
git clone https://github.com/Gnaroshi/GAIM.git
cd GAIM
export GAIM_CUDA_DEVICES=0,1,2,3
bash scripts/setup.sh
```

위 GPU 번호는 예시입니다. 자신의 할당 번호로 바꾸세요. 원래 서버는 4,5,6,7을 사용합니다. 설치 대상은 Ubuntu 22.04 x86_64, Python 3.10, 24GB GPU 네 장, CUDA 12.4 호환 드라이버입니다. `gcc`를 포함한 C 빌드 환경, `nvidia-smi`, `dpkg-deb`, 여유 공간 약 25GB와 인터넷이 필요합니다.

설치 파일은 프로젝트의 `.venv`, `.cache`, `.deps`, `data`에 둡니다. 시스템 Python은 바꾸지 않습니다. 환경만 먼저 확인하려면 `bash scripts/setup.sh --check`를 사용할 수 있습니다.

## 2. 어떤 실행이 필요한가요?

| 목적 | 할 일 |
|---|---|
| 저장된 결과만 확인 | 아래 CPU 검증 명령 실행 |
| GPU 메모리를 더 활용해 재학습 | [학습 안내](TRAINING_KO.md)의 준비 명령과 네 pane 명령 실행 |
| 원래 4비트 학습 설정 재현 | 아래 “기존 설정” 두 명령 실행 |
| 문장 생성부터 새로 시작 | 아래 “처음부터 생성” 네 명령 실행 |

같은 작업을 네 경로로 모두 실행할 필요는 없습니다. 새 실행 폴더 이름을 쓰세요.

**기록 확인 — GPU 불필요, 모델을 다시 실행하지 않음**

```bash
python3 scripts/verify_reference.py
```

정상 출력은 JSON의 `status: passed`입니다. 저장된 답변 재채점과 파일 확인 결과입니다.

**기존 설정 — 같은 입력으로 4비트 추가 학습 후 비교**

```bash
.venv/bin/python -m gaim.training \
  --source-run repro/reference/train --output-dir runs/reference_adapters \
  --steps 20 --allow-provisional
.venv/bin/python -m gaim.replay \
  --source-run repro/reference/test --training-dir runs/reference_adapters \
  --output-dir runs/reference_test --allow-provisional
```

학습 입력은 의료 문제 10개에서 만든 모델별 20개입니다. 모델별 갱신 횟수는 20회입니다. 평가에는 원문 100개와 문장을 붙인 입력 1,090개를 사용합니다. 공개 모델 그대로의 점수는 보존된 답변에서 가져오고, 새로 학습한 네 모델만 다시 답합니다.

**처음부터 생성 — 기존 생성·4비트 학습 설정 사용**

```bash
.venv/bin/python -m gaim.run \
  --config configs/train_generation.json --run-dir runs/new_train --limit 12
.venv/bin/python -m gaim.training \
  --source-run runs/new_train --output-dir runs/new_adapters --steps 20 --allow-provisional
bash scripts/run_evaluation.sh --run-dir runs/new_test
.venv/bin/python -m gaim.replay \
  --source-run runs/new_test --training-dir runs/new_adapters \
  --output-dir runs/new_test_replay --allow-provisional
```

새 문장이 달라지면 학습에 선택되는 문제 수도 달라질 수 있습니다. 원래 10개나 정확도에 맞추려고 입력을 바꾸지 않습니다. `--limit 12`, `--steps 20`을 생략하면 원래 작은 실험과 크기가 달라집니다.

## 3. 결과 읽기

학습은 `training.json`의 상태가 `complete`인지, 평가는 `COMPLETE`와 `report.md`가 있는지 확인합니다. 시작됐다는 표시만으로 완료로 세지 않습니다. 기존 결과는 [시험 결과](TEST_RESULTS_KO.md)에 있습니다.

`--allow-provisional`은 추가 문장의 의학적 무관성을 아직 검토하지 않은 상태로 탐색 실험을 한다는 뜻입니다. test 8문항에는 입력에 없는 그림도 언급됩니다. 수치를 발표할 때 이 조건을 함께 밝히세요.

동일한 seed라도 GPU·라이브러리에 따라 결과가 달라질 수 있습니다. 새 실행의 설정·문항 수·정확도와 차이를 그대로 기록합니다.

## 4. 실패하면

- **학습 실패:** 기존 폴더를 남기고 새 폴더에서 다시 시작합니다. 학습 중간 저장점에서 이어가는 기능은 없습니다.
- **생성 중단:** 코드·설정·입력·GPU 순서가 같으면 원래 생성 명령에 `--resume`을 붙입니다.
- **평가 중단:** 입력과 모델이 같으면 같은 replay 명령을 다시 실행합니다.
- **다른 OS/Python 또는 다운로드 checksum 오류:** 원인을 확인합니다. 검사 우회나 시스템 환경 덮어쓰기로 진행하지 않습니다.

## 5. Claude에게 전달할 요청

> CLAUDE.md를 읽어줘. 내가 사용할 GPU는 0,1,2,3이야. 새 학습을 준비하고 각 tmux pane에 붙여넣을 명령 네 개를 줘. 학습은 내가 시작할 테니 실행하거나 계속 모니터링하지 마. 각 모델이 어떤 추가 데이터로 학습되는지도 짧게 설명해줘.

GPU 번호는 본인 설정으로 바꾸세요. `CLAUDE.md`는 Claude Code의 프로젝트 지침 위치입니다. [공식 설명](https://code.claude.com/docs/en/memory)
