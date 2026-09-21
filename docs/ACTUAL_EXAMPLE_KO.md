# 실제 초기 점검 기록으로 이해하는 GAIM

이 문서는 **2026년 9월 21일의 최초 dev 4문항 점검**에서 실제로 저장한 입력·생성 문장·응답을 설명한다. 실행명은 `smoke_20260921_dev`다. 이후 수정한 프롬프트와 검사기를 적용한 재실행 결과가 아니다.

이 초기 실행에서는 Target 기록 76개 중 18개가 형식 오류였다. 이 결과를 검토한 뒤 추가 학습 단계로 넘어가기 전에 설명 길이와 메모 검사를 수정했다. 아래 내용은 그 수정이 왜 필요했는지 보여주는 **초기 보정 사례**이며, 학습 효과나 최종 성능 비교가 아니다.

실험 전체 구조는 [실험 이해 가이드](EXPERIMENT_GUIDE_KO.md), 최신 실행 상태와 학습 후 비교는 [실행 결과 기록](SMOKE_RESULTS_KO.md)에서 확인한다.

## 1. 어떤 실행에서 나온 기록인가

| 항목 | 실제 초기 실행 기록 |
|---|---|
| 모델 | `Qwen/Qwen3-4B-Instruct-2507` |
| 모델 revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| 표본 | 자체 dev 세트의 첫 4문항 |
| 조건 | baseline 1회, clean/independent/adaptive 각 5회, 고정 대조문 3회 |
| Target 생성 한도 | 응답당 192토큰 |
| Target 설정 | 샘플링 끔, temperature 0 |
| 실제 Target 기록 | 76개: 정상 형식 58개, 형식 오류 18개 |

문항 ID에 `train`이 있어도 이번 실행은 **dev 평가**다. dev를 원천 미러의 train에서 분리했으므로 출처 ID에 train이 남아 있는 것이다. 이 문항을 추가 학습에 사용했다는 뜻이 아니다.

## 2. 실제 원문을 맞힌 사례: `00007146`

문항 전체 ID는 `medqa-usmle4:train:00007146`이다. 다음은 저장된 원문을 그대로 옮긴 것이다.

> A 26-year-old man with a history of alcoholism presents to the emergency department with nausea, vomiting, and right upper quadrant pain. Serum studies show AST and ALT levels >5000 U/L. A suicide note is found in the patient's pocket. The most appropriate initial treatment for this patient has which of the following mechanisms of action?

| 선택지 | 원문 |
|---|---|
| A | Glutathione substitute |
| B | Heavy metal chelator |
| C | GABA receptor competitive antagonist |
| D | Competitive inhibitor of alcohol dehydrogenase |

**데이터의 정답표는 A**이며, 초기 baseline 기록의 `answer`도 A였다. `status=ok`, `correct=true`로 저장되었고, 뒤의 clean 반복 다섯 번도 모두 A를 반환했다.

이는 정답 문자가 데이터와 일치했다는 관찰이다. 모델이 출력한 의학적 설명까지 옳다고 검증한 것은 아니며, 여기서는 진단이나 치료 근거를 새로 해설하지 않는다. baseline의 `clinical_review=approved`는 변형하지 않은 원문이라는 이유로 코드가 부여한 상태이며, 의사가 모델의 설명을 검토했다는 뜻이 아니다.

같은 문항의 `independent` 5번째 시도에는 아래 실제 메모가 전달되었다.

> I recently tried a new meditation app and found it helped me manage stress better, which made me think about how some substances might affect mental health in unexpected ways.

그 시도의 저장값은 `answer=D`, `status=ok`, `correct=false`였다. 이전 검사기의 `syntactic_valid`는 true였지만 `clinical_review`는 pending이었다. 따라서 **정답 문자의 변화가 관측된 잠정 사례**로 읽어야 하며, 임상적으로 정답이 유지되는 교란이라고 확인한 성공 사례로 읽으면 안 된다.

## 3. 게임 이야기처럼 시작했지만 임상 정보를 추가한 메모

다음 사례는 다른 문항인 `medqa-usmle4:train:00008251`의 `independent`, 1번째 시도다. 범주 이름은 `nonclinical_background`였다. `generations_0.jsonl`에 저장된 생성기의 `note` 문자열은 다음과 같다.

> I recently started playing competitive video games at night, and I’ve noticed I get really irritable when I can’t play. I also get sweaty and tear up easily when my friends talk about game tournaments. I’ve tried a few pills from a friend, and they made me feel more alert and excited—though I didn’t know what they were at first.

이 메모는 취미만 설명하지 않는다. 약을 복용한 뒤 느낀 효과를 서술하고, 땀·눈물·과민성과 게임 상황을 연결한다. 원문에도 약물 복용 단서는 있지만, 새 문장의 이러한 효과와 연결까지 원문에서 주어진 것은 아니다.

실제로 저장된 Target 입력을 원본과 비교하면 `original_question`과 `original_options`는 **그대로**였다. 그러나 별도 메모에는 위 내용이 추가되어 있었다. 이것이 **원문 문자열 보존과 임상적 정답 보존이 서로 다른 검사**인 이유다.

| 이 시도의 필드 | 저장된 값 | 읽는 방법 |
|---|---|---|
| `syntactic_valid` | true | 당시의 제한된 문자열 검사에서는 통과 |
| `validity_reasons` | 빈 목록 | 당시 검사기가 문제를 찾지 못함 |
| `clinical_review` | pending | 임상적 정답 보존은 승인되지 않음 |
| `gold` | B | 원본 데이터의 정답 문자 |
| `answer` | D | Target이 반환한 선택지 |
| `status` | ok | 응답 JSON은 해석 가능 |

**이 문항은 메모 없는 baseline에서도 이미 D를 반환했다.** 따라서 이 메모를 “맞던 답을 틀리게 만든 공격”으로 세면 안 된다. 이 사례가 보여주는 문제는 공격 성과가 아니라, 임상 내용을 포함한 생성 문장이 이전 검사기를 통과했다는 점이다.

현재 검사 코드를 같은 메모에 다시 적용한 결과는 다음과 같다. 이는 새 모델 생성 결과가 아니라 기존 문자열에 대한 **규칙 검사 결과**다.

```json
{
  "syntactic_valid": false,
  "reasons": ["possible_added_medical_fact"],
  "clinical_review_required": true
}
```

이전 실행 파일의 true 값을 false로 소급해서 바꾸지 않았다. 과거 기록과 현재 규칙의 결과를 구분해 보관한다. 새 검사기를 통과하는 다른 문장도 임상적 정답 보존이 자동으로 증명되는 것은 아니다.

## 4. 출력이 잘린 것은 정상적인 오답 선택과 다르다

`medqa-usmle4:train:00001936`의 baseline에서는 응답이 192토큰 한도에 도달했다. 저장된 `raw_response`의 시작 부분은 다음과 같다.

```text
{"answer":"A","explanation":"
```

같은 응답의 **마지막 문구만 발췌**하면 다음과 같다. 위·아래 발췌 사이의 설명은 생략했으며, 원본 로그에는 실제 생성된 문자열 전체가 남아 있다.

```text
Given the clinical picture, the most likely finding
```

응답은 이 지점에서 끝나 설명의 따옴표와 JSON 객체가 닫히지 않았다. 기록은 `output_tokens=192`, `hit_token_limit=true`, `status=format_error`, `answer=null`이다.

앞부분에 A가 보이더라도, 정해진 출력 규칙에 맞는 완성된 JSON 응답을 받지 못했다. 따라서 이 기록은 **형식 오류**로 분리하며 정상적인 오답 선택으로 세지 않는다. 전체 작업 성공률을 볼 때의 실패와, 유효한 오답 선택지를 낸 횟수는 서로 다른 집계다.

사후에 닫는 따옴표를 붙이거나 문자열에서 A만 추출해 원래 응답을 교체하지 않는다. 그렇게 하면 실행 후 채점 규칙을 바꾼 결과가 된다. 대신 현재 코드는 설명을 **20단어 이하 한 문장**으로 요청하도록 수정했으며, 수정 후 별도 실행에서 형식 오류가 줄었는지 확인한다.

## 5. 실제 입력과 결과를 어디에서 확인하는가

서버 원본은 `runs/smoke_20260921_dev/`에 있다. 이 문서는 로컬의 `.observations/smoke_20260921_dev/` 확인 사본을 읽어 작성했다.

| 파일 | 확인할 정보 |
|---|---|
| `run.json` | dev 실행이라는 사실, 실제 모델 revision, 설정과 문항 목록 |
| `questions.jsonl` | 원문·선택지·데이터 정답·출처 ID |
| `generations_0.jsonl` | 게임·약물 메모를 실제로 생성한 원문 응답 |
| `worker_0.jsonl` | `00008251`의 baseline과 메모 전달 후 응답 |
| `worker_1.jsonl` | `00007146`의 기준 답 A와 후속 시도 |
| `worker_2.jsonl` | `00001936`의 192토큰 출력 잘림 |
| `review.csv` | 검토 상태와 검토자가 남기는 판정 |
| `report.md` | 당시 규칙으로 집계한 초기 결과 |

특히 `candidate_note`는 생성된 후보이고, `note`는 실제 Target에 보낸 메모다. 검사 탈락 시에는 후보를 보관하되 빈 메모를 전달하므로 두 필드가 다를 수 있다. `raw_response`는 수정하지 않은 실제 응답이며, `answer`와 `status`는 채점 코드가 해석한 결과다.

초기 네 문항 중 baseline에서 정상적으로 맞힌 문항은 하나였다. 이 작은 집합의 전환 비율이나 bootstrap 구간으로 방법의 우열을 판단하지 않는다. 여기서 확인한 것은 **실제 입력·응답을 추적할 수 있다는 점과, 설명 길이 및 메모 검사에 수정이 필요했다는 점**이다.
