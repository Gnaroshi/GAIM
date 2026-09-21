# GAIM — 의료 QA 교란 생성·평가·학습

의료 객관식 문제에 추가 정보를 넣었을 때 답이 바뀌는지 확인하고, 교란 데이터로 추가 학습한 모델이 같은 입력을 더 잘 견디는지 비교하는 대학원 수업 프로젝트입니다.

**Claude로 재현하려면 [CLAUDE.md](CLAUDE.md)를 먼저 읽으세요.** 환경 준비, 실행 순서와 예상 결과는 [재현 안내서](docs/REPRODUCING_KO.md)에 있습니다. 개인 서버 접속 권한이나 유료 AI API는 필요하지 않습니다. GPU 실행에는 본인이 사용할 수 있는 Linux GPU 서버가 필요합니다.

## 실제로 얻은 결과

2026-09-21의 고정 MedQA test 100문항과 기존 학습 10문항·20단계 모델을 평가했습니다.

| 모델 | 원문 100문항 정확도 | 같은 교란 입력 1,090건 정확도 |
|---|---:|---:|
| 추가 학습 전 | 66% | 65.7% |
| 원문만 학습 | 64% | 61.9% |
| 고정 배경 문장 학습 | 66% | 63.3% |
| 독립형 교란 학습 | 65% | 61.9% |
| 적응형 교란 학습 | 65% | 62.1% |

독립형과 적응형은 원래 맞힌 66문항 중 **같은 한 문항**에서만 잠정 오답 전환을 찾았습니다. 이번 작은 학습에서는 개선이 관측되지 않았습니다. 교란은 임상 미검토 상태이며, 이 결과는 의료 안전성이나 적응형 방법의 우월성을 입증하지 않습니다.

[전체 결과](docs/TEST_RESULTS_KO.md) · [실제 예시 6가지](docs/TEST_EXAMPLES_KO.md) · [누락 이미지 등 입력 점검](docs/TEST_DATA_NOTES_KO.md) · [개발 단계 기록](docs/SMOKE_RESULTS_KO.md)

## 먼저 기록부터 확인하기 — GPU 불필요

```bash
git clone https://github.com/Gnaroshi/GAIM.git
cd GAIM
python3 scripts/verify_reference.py
```

Python 3.10 이상에서 표준 라이브러리만 사용합니다. 저장된 입력과 답을 읽어 파일 무결성, 학습 입력 재구성, 실제 응답 재채점, 학습 전후 집계를 검증합니다. **모델을 새로 실행한 결과는 아닙니다.**

## GPU 실험 준비

검증한 설치 환경은 **Ubuntu 22.04 x86_64 / Python 3.10 / RTX 3090 24GB 네 장 / CUDA 12.4 호환 드라이버**입니다. 약 25GB 이상의 여유 공간과 다운로드 연결을 준비하세요. macOS·Windows·다른 Ubuntu/Python 조합의 GPU 설치는 자동 지원하지 않습니다. CPU 기록 검증은 별도입니다.

사용 권한이 있는 네 GPU를 명시합니다. 아래 0–3은 예시이며, 원래 실험은 4–7을 사용했습니다.

```bash
export GAIM_CUDA_DEVICES=0,1,2,3
bash scripts/setup.sh --check
bash scripts/setup.sh
```

`GAIM_CUDA_DEVICES` → 기존 `CUDA_VISIBLE_DEVICES` → 원래 기본값 `4,5,6,7` 순으로 선택합니다. 네 개의 중복 없는 숫자 ID만 지원합니다. 현재는 GPU 한두 장에서 순차 실행하는 모드를 구현하지 않았습니다. `cuda:0..3`은 선택한 네 장을 그 순서대로 다시 번호 붙인 것입니다. 다른 사람의 프로세스를 중단하지 마세요.

설치는 프로젝트의 `.venv`, `.cache`, `.deps`, `data`만 사용합니다. 원래 전체 의존성 버전을 제약으로 적용하고, 데이터·모델을 고정 revision에서 내려받아 검사합니다.

## 두 가지 재현 경로

### 교란 생성부터 전부 다시 실행

```bash
.venv/bin/python -u -m gaim.run \
  --config configs/train_generation.json --run-dir runs/reproduce_train --limit 12
.venv/bin/python -u -m gaim.training \
  --source-run runs/reproduce_train --output-dir runs/reproduce_adapters \
  --steps 20 --allow-provisional
bash scripts/run_evaluation.sh --run-dir runs/reproduce_test
.venv/bin/python -u -m gaim.replay \
  --source-run runs/reproduce_test --training-dir runs/reproduce_adapters \
  --output-dir runs/reproduce_test_replay --allow-provisional
```

원래 설정을 다시 실행하지만, GPU 계산·라이브러리 차이에 따라 생성 문구와 학습 데이터가 달라질 수 있습니다. `--limit 12`와 `--steps 20`을 생략하면 다른 실험이 됩니다. 새 실행 이름을 쓰고 결과 차이를 그대로 보고하세요.

### 원래 생성했던 정확한 입력으로 학습·평가

```bash
.venv/bin/python -u -m gaim.training \
  --source-run repro/reference/train --output-dir runs/fixed_input_adapters \
  --steps 20 --allow-provisional
.venv/bin/python -u -m gaim.replay \
  --source-run repro/reference/test --training-dir runs/fixed_input_adapters \
  --output-dir runs/fixed_input_test_replay --allow-provisional
```

이 경로는 저장된 생성 기록을 재사용합니다. 원래의 공통 학습 10문항·조건별 20행과 test 입력을 재구성하며, 생성 모델을 다시 실행하지 않습니다. 기반 모델 점수는 보존된 원래 응답에서 집계하고, 새로 학습한 네 모델만 다시 추론합니다. 학습 가중치가 비트까지 같아지는 것은 보장하지 않습니다.

`--allow-provisional`은 미검토 교란을 포함한 탐색 실험임을 명시합니다. 자동 검사 통과와 임상적 정답 보존 승인은 다릅니다. 이미지가 빠진 test 문항 8개를 포함하며, 사후에 유리한 문항만 골라 결과를 만들지 않습니다.

## 보존된 것과 다시 만드는 것

| 포함 | 위치 |
|---|---|
| 실행 코드·고정 설정·소프트웨어 테스트 | `gaim/`, `configs/`, `tests/` |
| 생성 입력·실제 응답·검토 상태·기준 결과 | `repro/reference/` |
| 원래 모델 revision·환경·문항 ID·해시 | `docs/results/reference/` |
| 한국어 실험 설명과 예시 | `docs/` |

모델 가중치, adapter 가중치, 가상환경, 캐시, 개인 경로와 프로세스 로그는 Git에 넣지 않았습니다. 새 clone에는 학습된 adapter가 없으므로 학습을 끝낸 뒤 replay를 실행해야 합니다. 공개한 데이터와 모델의 출처·라이선스는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 있습니다.

`reference-20260921` 태그는 원래 실행 소스를 보존합니다. `main`은 [운영 변경 기록](docs/PORTABILITY.md)에 설명한 GPU 선택·설치·검증 개선을 포함합니다. 기록의 원래 코드 해시를 현재 코드 해시라고 바꾸지 않습니다.

## 결과와 문제 해결

실행 폴더의 `COMPLETE`, `report.md`, `summary.json`, `run.json` 또는 `training.json`을 확인하세요. 실패·부분 완료 기록은 보존하고, [재현 안내서](docs/REPRODUCING_KO.md)의 중단·재개 절차를 따르세요. 기록을 수정하거나 재시도 중 가장 좋은 결과만 고르면 재현이 아닙니다.

```bash
.venv/bin/python -m pytest -q
```

소프트웨어 검사는 가상 fixture와 공개 기록 무결성 검사를 포함하며, 그 통과 수를 의료 연구 관측값으로 세지 않습니다. 자세한 방법은 [실험 이해 가이드](docs/EXPERIMENT_GUIDE_KO.md), 코드 대응은 그 문서의 파일 안내를 참고하세요.
