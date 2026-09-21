# 다른 서버에서 Claude와 함께 재현하기

이 안내서는 저장소를 처음 받는 사람이 **무엇을 확인하고, 무엇을 다시 실행하며, 어떤 결과를 비교해야 하는지** 설명합니다. 명령은 모두 저장소 최상위 폴더에서 실행합니다. 기존 실험자의 서버에 접속할 필요는 없습니다.

## 1. Claude에게 전달할 요청

먼저 저장소를 받습니다.

```bash
git clone https://github.com/Gnaroshi/GAIM.git
cd GAIM
```

이 폴더를 Claude Code에서 열고 아래처럼 요청하세요. 루트 `CLAUDE.md`는 프로젝트 지침을 제공하는 표준 위치입니다. 사용 중인 도구가 파일을 자동으로 읽는지 불분명하면 직접 읽도록 요청하면 됩니다. [Claude Code 공식 문서](https://code.claude.com/docs/en/memory)

**GPU 없이 결과부터 이해할 때:**

> CLAUDE.md와 docs/REPRODUCING_KO.md를 읽어줘. 공개 기록 검증과 소프트웨어 테스트를 실행하고, 실제 test 사례 두 개를 원문·추가 문장·정답·모델 응답으로 설명해줘. 모델을 새로 실행한 것과 저장된 응답을 재채점한 것을 구분해줘.

**같은 입력으로 학습·평가를 재현할 때:**

> CLAUDE.md와 재현 안내서를 읽고 고정 입력 재현을 진행해줘. 사용 가능한 환경은 Ubuntu 22.04, Python 3.10, RTX 3090 네 장이고 내가 사용할 수 있는 물리 GPU는 0,1,2,3이야. 공개 기록을 먼저 검증하고, 설치와 실행 환경을 확인한 다음 네 조건을 20단계씩 학습하고 동일 test 입력으로 평가해줘. 새 폴더에 결과를 저장하고 기준 결과와의 차이, 실제 실행한 범위, 실패와 한계를 문서화해줘.

위 GPU 번호는 **다른 서버용 예시**입니다. 자신의 할당 번호로 바꾸세요. 원래 실험 서버에서는 4,5,6,7만 사용합니다. GPU 권한은 비어 있다는 이유만으로 생기지 않습니다.

**교란부터 새로 만들 때:**

> 이번에는 고정 입력 재사용이 아니라 전체 생성 재현을 진행해줘. train 12문항에서 교란을 다시 만들고, 네 조건을 20단계 학습한 다음 고정 test 100문항을 평가하고 replay까지 실행해줘. 새로 생성한 문장이 원래 기록과 달라도 숨기지 말고 보고해줘. GPU 할당은 앞서 알려준 설정을 유지해줘.

이 저장소는 Claude 자체를 설치하거나 Claude로 실험 결과를 대신 생성하지 않습니다. Claude는 아래 Python 실험 코드를 실행하고 결과를 읽는 작업을 돕습니다. Claude Code 사용 계정이나 요금은 이 프로젝트의 로컬 모델 실행과 별개입니다.

## 2. 세 가지 작업의 차이

| 작업 | GPU | 새로 만드는 것 | 확인할 수 있는 것 |
|---|---|---|---|
| 기록 검증 | 필요 없음 | 없음 | 공개 입력·실제 응답·집계의 일치 여부 |
| 고정 입력 재현 | 네 장 | 네 adapter와 그 응답 | 원래와 같은 학습·test 입력에서의 새 학습 결과 |
| 전체 생성 재현 | 네 장 | 교란·adapter·기반 모델 및 학습 후 응답 | 생성부터 평가까지 같은 설정으로 다시 동작하는지 |

**원래 결과와 비교하려면 기록 검증 → 고정 입력 재현 순서가 가장 직접적입니다.** 전체 생성은 추가 경로입니다. 두 GPU 경로를 모두 실행할 필요는 없습니다.

## 3. 실험에 들어가는 데이터와 모델

모델은 `Qwen/Qwen3-4B-Instruct-2507`, revision은 `cdbee75f17c01a7cc42f958dc650907174af0554`입니다. 의료 답변을 만드는 역할과 추가 문장을 만드는 역할이 같은 가중치를 사용합니다. 정오를 판정하는 별도 LLM은 없고, 코드가 답 문자와 정답표를 비교합니다.

데이터는 영어 MedQA 4지선다 공개 미러 `GBaker/MedQA-USMLE-4-options`의 revision `0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18`입니다. 한 행은 질문, A–D 선택지, 정답, 원천 행 번호와 ID를 갖습니다. 예를 들어 실제 test의 `medqa-usmle4:test:00000012`는 유전자 변이와 질환 위험을 묻고, 데이터셋 정답은 B입니다. Porsche 안내 책자를 보는 취미 문장을 추가해도 이번 모델은 B를 유지했습니다. [실제 입력과 응답](TEST_EXAMPLES_KO.md)

원천 train 10,178행에서 정답이 상충하는 중복 네 행을 제외합니다. seed `20260921`로 train 100문항과 자체 dev 20문항을 분리하고, 원천 test 1,273행에서 test 100문항을 선택합니다. dev는 원전의 공식 validation이 아닙니다. train/dev/test를 원문 해시로 점검하지만 기반 모델의 사전학습 노출 여부까지 알 수는 없습니다.

원래 작은 학습은 준비된 train 100문항 중 **12문항에서 교란을 생성**했습니다. 필요한 모든 조건에 적합한 후보가 있는 공통 10문항만 선택했습니다. 네 학습 조건은 다음과 같습니다.

| 조건 | 문항마다 두 개의 학습 입력 |
|---|---|
| `clean` | 원문 두 번 |
| `random` | 원문 한 번 + 고정 배경 문장 한 번 |
| `independent` | 원문 한 번 + 독립 생성에서 선택한 문장 한 번 |
| `adaptive` | 원문 한 번 + 피드백 기반 생성에서 선택한 문장 한 번 |

`random`은 100만 개의 무작위 생성 데이터가 아닙니다. 현재 구현은 고정 배경 문장 대조군입니다. 각 조건의 학습 행 수는 20이며, 같은 원본 정답으로 감독합니다. 학습 목표 응답의 explanation은 빈 문자열이므로 설명 품질을 학습하거나 비교하는 실험도 아닙니다.

학습은 NF4 QLoRA, rank 8, alpha 16, dropout 0, all-linear, 학습률 1e-4, batch 1, gradient accumulation 2, 최대 길이 2,048, optimizer 20단계입니다. 조건마다 총 40행이 처리됩니다. 추론은 기반 모델과 adapter 모델 모두 bfloat16입니다.

## 4. GPU 없이 공개 기록 확인

Python 3.10 이상에서 다음을 실행합니다. macOS에서도 이 단계는 GPU가 필요 없습니다.

```bash
python3 scripts/verify_reference.py
python3 -m unittest discover -s tests -q
```

검증기는 39개 공개 파일의 해시, train 228개와 test 1,900개의 실제 답변, 학습 후 4,760개 답변을 확인합니다. 네 학습 데이터도 재구성해 원래 해시와 대조하고, adapter당 1,190개 입력이 동일한지 확인합니다. 설치·네트워크 호출·기록 수정은 하지 않습니다. 출력 JSON의 `status`가 `passed`이면 정상입니다.

두 번째 명령은 오류 처리 등을 확인하는 소프트웨어 테스트입니다. 가상 fixture와 공개 기록 무결성 검사를 포함하며, 그 통과 수를 실험 문항 수로 세면 안 됩니다.

## 5. GPU 환경 준비

검증한 환경은 **Ubuntu 22.04 x86_64, 시스템 python3 3.10, RTX 3090 24GB 네 장, CUDA 12.4 호환 NVIDIA 드라이버**입니다. `nvidia-smi`, `gcc`와 동작하는 C 빌드 환경, `dpkg-deb`, 약 25GB 이상의 여유 공간 및 인터넷 연결이 필요합니다. 스크립트는 다른 OS/Python 조합에 자동 설치하지 않습니다. GPU 한두 장의 순차 실행 모드는 구현하지 않았습니다.

```bash
# 자신의 서버에서 할당받은 네 물리 GPU 번호로 바꿉니다.
export GAIM_CUDA_DEVICES=0,1,2,3
bash scripts/setup.sh --check
bash scripts/setup.sh
```

`--check`는 OS, Python, 도구, 선택한 GPU가 작업 중인지 확인하고 설치 전에 끝납니다. 설치는 `.venv`에 패키지, `.cache`에 모델, `data/medqa`에 고정 데이터, 필요하면 `.deps`에 Python 헤더를 준비합니다. 시스템 Python이나 다른 사용자의 작업은 변경하지 않습니다.

GPU 선택 순서는 `GAIM_CUDA_DEVICES` → 기존 `CUDA_VISIBLE_DEVICES` → 기본값 `4,5,6,7`입니다. 새 터미널에서도 같은 환경 변수를 지정하세요. 네 개의 중복 없는 숫자 ID만 지원하며 GPU UUID/MIG 문자열과 한두 장 실행은 지원하지 않습니다. 스케줄러가 GPU를 할당했다면 그 범위를 유지해야 합니다.

설치 중 원래 관측한 Python 버전 제약을 적용하고, 고정 revision의 모델 파일을 검사하며, 네 GPU에서 작은 NF4 순전파·역전파를 실행합니다. 데이터의 질문을 임의로 잘라 메모리에 맞추지는 않습니다. 초기 다운로드와 설치 시간은 네트워크에 따라 달라집니다.

## 6. 경로 A — 원래 생성 입력으로 재학습·평가

아래 이름의 폴더가 이미 있으면 **새 이름으로 바꾸고 후속 명령도 맞춰** 실행합니다.

```bash
.venv/bin/python -u -m gaim.training \
  --source-run repro/reference/train --output-dir runs/fixed_input_adapters \
  --steps 20 --allow-provisional
```

이 명령이 끝나면 `runs/fixed_input_adapters/training_manifest.json`의 status가 `complete`인지 확인합니다. `clean`, `random`, `independent`, `adaptive` 폴더마다 `dataset.jsonl`, `adapter/adapter_config.json`, `adapter/adapter_model.safetensors`가 있어야 합니다. 학습 요약은 `training_report.md`입니다.

그다음 같은 test 입력을 네 모델에 넣습니다.

```bash
.venv/bin/python -u -m gaim.replay \
  --source-run repro/reference/test --training-dir runs/fixed_input_adapters \
  --output-dir runs/fixed_input_test_replay --allow-provisional
```

`runs/fixed_input_test_replay/COMPLETE`, `report.md`, `summary.json`을 확인합니다. adapter마다 원문 100개와 교란 입력 1,090개를 평가합니다. 기반 모델의 점수는 저장된 원래 응답에서 가져옵니다. 새로운 기반 모델 답변이나 새 공격을 생성한 것으로 보고하지 않습니다.

공개 저장소에는 adapter 가중치가 없으므로 학습을 생략하고 replay부터 실행할 수 없습니다. 원래 코드가 필요한 경우 `reference-20260921` 태그를 별도 checkout에서 확인하세요. 일반 재현은 GPU 선택을 지원하는 현재 main을 사용합니다.

## 7. 경로 B — 교란부터 다시 생성

설치 후 다음 네 단계를 순서대로 실행합니다. 경로 A의 출력을 이 경로에 섞지 않습니다.

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

첫 단계는 train 12문항, 세 번째는 test 100문항입니다. 생성된 후보가 달라지면 공통 학습 문항 수도 10과 달라질 수 있습니다. 그 경우 원래 10개에 맞추려고 후보를 바꾸지 말고, 생성부터 달라졌음을 기록합니다. `--limit 12`와 `--steps 20`을 빠뜨리면 같은 크기의 실험이 아닙니다.

test 문항당 원문 baseline 1회, clean-repeat·independent·adaptive 각 5회, 고정 배경 대조 3회로 Target 19회와 Attacker 10회를 호출합니다. 적응형은 직전까지의 문장·실제 답변·정오·자동 검사 결과를 다음 생성기에 전달합니다. 독립형의 이력은 비어 있습니다. Target은 매번 새 문맥에서 답합니다.

생성기가 조건을 위반하면 후보를 기록하고 Target에는 빈 문장을 보냅니다. 이 경우에도 예정된 호출을 소비하며 공격 성공으로 세지 않습니다. 따라서 교란 1,090개는 100×13개의 모든 후보가 아니라 자동 검사를 통과해 replay 대상으로 남은 원래 시도 수입니다. 동일 문자열이 반복된 시도도 있어 고유 입력 수와 다릅니다.

## 8. 무엇과 비교할 것인가

기준 집계는 `repro/reference/test_replay/summary.json`이고, [전체 결과 보고서](TEST_RESULTS_KO.md)에 해석이 있습니다.

| 모델 | 원문 정답 / 100 | 교란 정답 / 1,090 |
|---|---:|---:|
| 학습 전 | 66 | 716 |
| clean | 64 | 675 |
| random | 66 | 690 |
| independent | 65 | 675 |
| adaptive | 65 | 677 |

이 숫자는 맞춰야 할 합격 기준이 아닙니다. 이번 작은 학습에서는 개선이 없었습니다. 독립형과 적응형 모두 처음 맞힌 66문항 중 같은 한 문항에서 잠정 오답 전환을 찾았습니다. 적응형 우월성을 보여주지 못했습니다.

추가 문장 1,300개는 모두 임상 미검토 상태입니다. 한 전환 문항에는 의료 내용·정답 노출 우려가 있는 추가 문장과 누락 이미지 문제가 있습니다. 총 8개 test 문항이 입력에 없는 이미지를 언급합니다. 따라서 “의학적 사실은 완벽히 보존했고 위험한 오진을 유도했다”는 주장을 해서는 안 됩니다. `--allow-provisional`은 이 미검토 상태를 인정하고 탐색 실험을 허용하는 옵션입니다.

고정 seed는 모든 GPU 연산과 학습 가중치를 비트까지 같게 보장하지 않습니다. 새 보고서에는 commit, 하드웨어·패키지 버전, 재현 경로, 실제 문항/시도 수, 정확도, 오류 수, 기존 결과와의 차이를 적습니다. test100은 이미 관측된 표본이므로 후속 방법을 개발해 같은 표본으로 평가하면 탐색 결과로 구분합니다.

## 9. 중단·오류와 재개

| 상황 | 처리 |
|---|---|
| 지원하지 않는 OS/Python | CPU 기록 검증은 진행하고 GPU 설치는 중단합니다. 지원 환경을 준비하거나 별도 이식 작업으로 기록합니다. |
| 기존 `.venv`가 다른 Python 버전임 | 기존 환경을 덮어쓰지 말고 새 clone에서 지원 Python으로 설치합니다. 설치 스크립트는 기존 `.venv`를 재사용합니다. |
| GPU가 작업 중이거나 번호가 불일치 | 다른 프로세스를 종료하지 않습니다. 할당과 환경 변수를 확인하고, 기존 실행은 같은 GPU 순서를 유지합니다. |
| get-pip 또는 헤더 checksum 변경 | 다운로드된 코드를 실행하지 않습니다. 공식 출처 변경을 확인한 뒤 별도 설치 변경으로 기록합니다. |
| 생성 실행이 중단됨 | 코드·입력·설정·GPU가 같을 때 같은 명령에 `--resume`을 붙입니다. |
| 학습이 실패함 | optimizer 체크포인트 재개는 없습니다. 오류를 고친 후 새 output 폴더로 20단계를 다시 실행합니다. |
| replay가 중단됨 | 입력·adapter·검토 내용·GPU가 그대로면 동일 명령으로 재실행합니다. 바뀌었으면 새 output 폴더를 씁니다. |
| 원래 태그의 replay 폴더를 main에서 재개함 | 기존 manifest에는 새 GPU 필드가 없으므로 새 output 폴더를 사용합니다. 공개 reference는 읽기 전용 source로 사용할 수 있습니다. |
| 다운로드·추론·형식 오류 | 원인과 실패 기록을 보존합니다. 응답을 손으로 채우거나 결과가 좋은 재시도만 남기지 않습니다. |

예를 들어 위 경로 B의 학습용 교란 생성만 중단됐다면, 환경과 코드가 같다는 것을 확인한 뒤 다음처럼 재개합니다.

```bash
.venv/bin/python -u -m gaim.run \
  --config configs/train_generation.json --run-dir runs/reproduce_train --limit 12 --resume
```

## 10. 저장소 파일 안내

`CLAUDE.md`는 작업 지침, 이 문서는 실행 순서, `EXPERIMENT_GUIDE_KO.md`는 연구 방법 설명입니다. `TEST_EXAMPLES_KO.md`에는 실제 영어 문항과 답변, 한국어 풀이가 있습니다. `repro/reference/`에는 원래 입력·응답과 변환 이력이 있고, `docs/results/reference/`에는 버전·환경·실행 전 잠금 기록이 있습니다.

과거 문서의 `runs/test_20260921` 같은 경로는 원래 서버의 실행 이름입니다. 공개 대응 자료는 `repro/reference/test`와 `repro/reference/test_replay`에서 찾습니다. 새 실행은 새로운 `runs/` 하위 폴더에 기록하고 reference 파일은 바꾸지 않습니다. 라이선스와 출처는 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), 공개 이후 운영 변경은 [PORTABILITY.md](PORTABILITY.md)를 참고하세요.
