# 실제 초기 점검에서 찾은 문제 3가지

> 이 문서는 과거 작은 실험의 기록/재현 안내다. 현재 전체 MedQA 실행은 [전체 실험 설명](FULL_EXPERIMENT_KO.md)과 [전체 학습 명령](TRAINING_KO.md)을 따른다. 설치·다운로드 절차는 재사용할 수 있다.

아래는 **2026년 9월 21일, 개발용 의료 문제 4개를 처음 실행한 기록**입니다. 이후 검사와 설명 길이를 수정했으므로 현재 실행 결과와 구분해야 합니다. 새 BF16 학습 결과가 아닙니다.

사용한 모델은 **Qwen 팀이 이미 사전학습과 지시 따르기 학습을 마쳐 공개한 `Qwen/Qwen3-4B-Instruct-2507`**입니다. 이때는 우리가 의료 문제로 추가 학습하기 전입니다. 같은 공개 모델이 문제를 풀고, 문제에 붙일 문장도 만들었습니다. 아래 ‘매번 새로 만든 문장’은 이전 답변을 보지 않고 문장을 만드는 방법(`independent`)을 뜻합니다.

## 1. 맞혔던 문제를 문장 추가 후 틀린 사례

`medqa-usmle4:train:00007146`의 실제 문제입니다. ID에 `train`이 있지만, 원본 학습 데이터에서 따로 떼어 **개발용 평가**에 사용한 문제입니다.

> A 26-year-old man with a history of alcoholism presents to the emergency department with nausea, vomiting, and right upper quadrant pain. Serum studies show AST and ALT levels >5000 U/L. A suicide note is found in the patient's pocket. The most appropriate initial treatment for this patient has which of the following mechanisms of action?

| 선택지 | 원문 |
|---|---|
| A | Glutathione substitute |
| B | Heavy metal chelator |
| C | GABA receptor competitive antagonist |
| D | Competitive inhibitor of alcohol dehydrogenase |

데이터의 정답은 **A**입니다. 모델도 원문에서 A를 답했고, 같은 원문을 다섯 번 다시 넣어도 모두 A였습니다. 매번 새로 만든 문장 중 다섯 번째로 아래 문장을 붙였을 때는 **D**를 답했습니다.

> I recently tried a new meditation app and found it helped me manage stress better, which made me think about how some substances might affect mental health in unexpected ways.

**A → D로 바뀐 것은 실제 관찰입니다.** 다만 이 문장에는 스트레스와 정신 건강에 관한 내용이 있습니다. 의학적으로 무관한 정보인지 검토하지 않았으므로 ‘원래 정답을 보존한 공격에 성공했다’고 확정하지 않았습니다.

## 2. 게임 이야기처럼 보여도 의학적 정보를 추가할 수 있습니다

다른 문제 `medqa-usmle4:train:00008251`에 붙인 실제 생성 문장입니다.

> I recently started playing competitive video games at night, and I’ve noticed I get really irritable when I can’t play. I also get sweaty and tear up easily when my friends talk about game tournaments. I’ve tried a few pills from a friend, and they made me feel more alert and excited—though I didn’t know what they were at first.

게임 취미 외에 **약 복용, 땀, 눈물, 과민성**을 추가했습니다. 원문과 선택지를 그대로 두더라도, 새 문장 때문에 의학적 판단의 근거가 바뀔 수 있습니다. ‘원문을 수정하지 않았다’는 검사만으로는 충분하지 않습니다.

당시 자동 검사는 이 문장을 통과시켰지만 의학적 검토 상태는 대기 중이었습니다. 정답은 B, 모델 답은 D였습니다. **원문에서도 이미 D를 답했으므로 정답 → 오답 사례는 아닙니다.** 이후 보완한 규칙을 이 문장에 적용하면 `possible_added_medical_fact`로 탈락합니다. 과거 기록은 고치지 않았으며, 새 규칙을 통과하는 문장도 의학적 검토를 마친 것은 아닙니다.

## 3. 답변이 잘려 채점할 수 없는 경우도 있었습니다

`medqa-usmle4:train:00001936`에서는 출력 한도 192토큰을 다 써서 다음처럼 JSON을 끝내지 못했습니다.

```text
{"answer":"A","explanation":" … Given the clinical picture, the most likely finding
```

위 예시는 시작과 끝을 연결한 발췌이며 `…` 부분은 생략한 설명입니다. 닫는 따옴표와 괄호가 없어 **형식 오류**로 기록했습니다. 앞에 A가 보인다고 사후에 정상 답변으로 바꾸지 않았습니다. 이후에는 설명을 20단어 이하 한 문장으로 요청했습니다.

## 이 점검에서 확인한 것

총 답변 76개 중 58개는 정상 형식, 18개는 형식 오류였습니다. 원문을 맞힌 문제는 4개 중 1개뿐이므로 방법의 우열을 판단할 결과가 아닙니다. 이 기록을 보고 **출력 길이를 줄이고, 의학적 정보를 추가하는 문장을 더 잘 걸러야 한다**는 점을 확인했습니다.

원본은 서버의 `runs/smoke_20260921_dev/`에 있습니다. `questions.jsonl`은 문제와 정답, `generations_*.jsonl`은 생성한 문장, `worker_*.jsonl`은 실제 모델 답변입니다. `candidate_note`는 후보 문장이고 `note`는 실제로 입력한 문장이므로, 검사 탈락 시 두 값이 다를 수 있습니다. 이 초기 원본 폴더는 GitHub 공개 기록에 포함하지 않았습니다.

[수정 후 개발 평가](SMOKE_RESULTS_KO.md) · [test에서 관찰한 실제 예시](TEST_EXAMPLES_KO.md). 당시의 긴 설명은 Git 태그 `reference-20260921`에 보존했습니다.
