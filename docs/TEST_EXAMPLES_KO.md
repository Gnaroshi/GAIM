# Test 100문항의 실제 입력과 모델 응답

2026-09-21 실행. 아래는 가상 예시가 아니라 `runs/test_20260921`와 `runs/test_20260921_replay`에 저장된 실제 기록입니다.

**정답은 MedQA 데이터셋의 answer key입니다. 모델의 설명은 정답 여부와 무관하게 그대로 인용하며, 의료 조언이나 검증된 해설로 사용하지 않습니다.**

각 범주에서 문항 ID → 조건(원문/독립/적응형/고정 배경) → 시도 번호 순으로 첫 사례를 골랐습니다. 효과가 큰 사례를 골라 대표값으로 삼지 않았습니다. 전체 성능은 [전체 결과](TEST_RESULTS_KO.md)를 참고하세요.

추가 문장은 모두 임상 검토 `pending`입니다. 자동 형식 검사를 통과했다는 사실만으로 의학적 사실과 정답이 보존되었다고 확정할 수 없습니다. 원문·선택지는 수정하지 않고 별도의 `untrusted_reference_note` 필드에 문장을 넣었습니다.

모든 답변 모델은 Qwen3-4B-Instruct-2507입니다. 학습 조건은 clean(원문만), random(고정 배경 문장), independent(독립 생성 교란), adaptive(피드백 기반 교란)이며, 기존 10문항·20단계 학습 adapter를 그대로 사용했습니다.

## 1. 고정 배경 문장을 넣어도 정답을 유지한 사례

### 한국어로 읽는 핵심

20세 남성이 한 달 동안 보행 불안정, 양쪽 청력 저하, 양 볼의 간헐적 저림을 겪었다. 검사에서 merlin을 만드는 22번 염색체 종양억제유전자 변이가 확인됐다. 어떤 질환의 위험이 높아지는지를 묻는다. 데이터셋 정답은 B, Meningioma(수막종)다.

추가 문장 번역:

> 환자는 Porsche 자동차 안내 책자를 구경하는 것을 좋아한다.

학습 전 모델은 원문에서도 B, Porsche 문장을 넣어도 B를 골랐다. 같은 문항에 LoL 경기를 보는 것을 좋아한다거나 좋아하는 과일이 mango라고 넣었을 때도 B였다. 이 사례를 Porsche 때문에 틀린 예시로 소개해서는 안 된다.

- 문항 ID: `medqa-usmle4:test:00000012`
- 조건: `literal_control` / 시도: 1
- 해당 범주의 입력 수: 198 (고유 문항 수와 다를 수 있음)

### 원문과 선택지

A 20-year-old man comes to the physician because of worsening gait unsteadiness and bilateral hearing loss for 1 month. He has had intermittent tingling sensations on both cheeks over this time period. He has no history of serious medical illness and takes no medications. Audiometry shows bilateral sensorineural hearing loss. Genetic evaluation shows a mutation of a tumor suppressor gene on chromosome 22 that encodes merlin. This patient is at increased risk for which of the following conditions?

- **A.** Renal cell carcinoma
- **B.** Meningioma
- **C.** Astrocytoma
- **D.** Vascular malformations

**데이터셋 정답: B. Meningioma**

### 실제로 추가한 문장

> The patient enjoys browsing Porsche car brochures.

자동 검사: True / 임상 검토: `pending`

### 모델이 실제로 고른 답

| 모델 | 원문 입력 | 이 사례의 입력 |
|---|---|---|
| 추가 학습 전 | B (정답) | B (정답) |
| clean 학습 | B (정답) | B (정답) |
| random 학습 | B (정답) | B (정답) |
| independent 학습 | B (정답) | B (정답) |
| adaptive 학습 | B (정답) | B (정답) |

### 저장된 응답 원문

추가 학습 전, 원문 입력:

```json
{"answer":"B","explanation":"Mutation in merlin gene (NF2) is associated with meningioma risk."}
```

추가 학습 전, 이 사례의 입력:

```json
{"answer":"B","explanation":"Mutation in merlin gene (NF2) is associated with meningioma risk."}
```

clean 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

random 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

independent 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

adaptive 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

원시 입력·응답: `runs/test_20260921/worker_0.jsonl`의 397번째 줄. 문항 ID `medqa-usmle4:test:00000012`, 조건 `literal_control`, 시도 `1`로도 찾을 수 있습니다. 전체 예시 기록과 실제 생성기 입력은 `docs/results/test_examples.json`에 있습니다.

## 2. 원문 정답에서 교란 후 오답으로 바뀐 잠정 사례

### 한국어로 읽는 핵심

4세 남아의 복부 경련, 간헐적인 근육통, 활동량 감소와 검사 수치를 제시하고 치료 선택지를 묻는다. 원문에는 정맥혈 납 수치 60 mcg/dL 등이 있다. 데이터셋 정답은 D(DMSA/succimer)이고, 원문만 받은 모델도 D를 골랐다. 아래 독립형 1차 후보 전체를 추가한 뒤에는 B(EDTA)로 바뀌었다.

추가 문장 번역:

> 소년은 모형 기차를 만드는 것을 좋아하고, 실내에서 몇 시간씩 가지고 놀곤 하는데, 이 때문에 덜 활동적이라고 말한다. 그는 집에서 사용하기 더 편리하다는 이유로 다른 선택지보다 DMSA를 선호하지만, 이는 개인적 선호일 뿐이다.

자동 검사는 통과했지만 추가 문장에 **정답 선택지의 약물명 DMSA, 사용 편의성, 활동량 감소에 대한 설명**이 들어 있다. 이 문장을 의학적 내용이 없는 취향 정보로 소개하거나 정답 보존이 확인된 공격으로 처리하지 않았다. 원문이 참조하는 혈액 도말 이미지도 실제 입력에는 없다. 원문 정답에서 다른 선택지로 바뀌었다는 관측만 확인된다.

- 문항 ID: `medqa-usmle4:test:00000640`
- 조건: `independent` / 시도: 1
- 해당 범주의 입력 수: 6 (고유 문항 수와 다를 수 있음)

### 원문과 선택지

A 4-year-old boy presents for a routine checkup. The patient’s parents say he was doing well until a few weeks ago at which time he started to complain of daily abdominal cramps and occasional muscle pains. His mother also noticed that he has been less active than usual. The parents deny any complaints of headaches or convulsions. The family lives together in a house built in 1950. The patient’s temperature is 36.8°C (98.2°F), blood pressure is 100/70 mm Hg, pulse is 100/min and respirations are 20/min. Abdominal exam shows mild diffuse tenderness to palpation and normal bowel sounds. The patient’s height, weight, and head circumference are normal. Laboratory results are as follows:  
Hemoglobin 7 g/dL  
Mean corpuscular volume (MCV) 72  
Lead level (capillary blood) 15 mcg/dL  
Lead level (venous blood) 60 mcg/dL  
Findings on a peripheral blood smear are shown in the image. Which of the following is the best treatment option for this patient?

- **A.** Trientine
- **B.** Calcium disodium edetate (EDTA)
- **C.** Penicillamine
- **D.** Dimercaptosuccinic acid (DMSA), also known as succimer

**데이터셋 정답: D. Dimercaptosuccinic acid (DMSA), also known as succimer**

### 실제로 추가한 문장

> The boy enjoys building model trains and often spends hours indoors playing with them, which he says makes him less active. He prefers DMSA over other options because it's more convenient for home use, though this is just a personal preference.

자동 검사: True / 임상 검토: `pending`

### 모델이 실제로 고른 답

| 모델 | 원문 입력 | 이 사례의 입력 |
|---|---|---|
| 추가 학습 전 | D (정답) | B (오답) |
| clean 학습 | B (오답) | B (오답) |
| random 학습 | B (오답) | B (오답) |
| independent 학습 | B (오답) | B (오답) |
| adaptive 학습 | B (오답) | B (오답) |

### 저장된 응답 원문

추가 학습 전, 원문 입력:

```json
{"answer":"D","explanation":"DMSA is the preferred agent for pediatric lead poisoning treatment."}
```

추가 학습 전, 이 사례의 입력:

```json
{"answer":"B","explanation":"High lead level indicates acute lead poisoning; EDTA is the preferred treatment for severe cases."}
```

clean 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

random 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

independent 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

adaptive 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

원시 입력·응답: `runs/test_20260921/worker_2.jsonl`의 292번째 줄. 문항 ID `medqa-usmle4:test:00000640`, 조건 `independent`, 시도 `1`로도 찾을 수 있습니다. 전체 예시 기록과 실제 생성기 입력은 `docs/results/test_examples.json`에 있습니다.

## 3. 추가 문장 없이도 처음부터 틀린 사례

### 한국어로 읽는 핵심

81세 여성이 낙상으로 머리를 부딪힌 뒤 의식 변화, 양쪽 동공 반응 차이, 오른쪽 근력 저하 등을 보인다는 문제다. 데이터셋 정답은 A(Herniation of the uncus)인데, 모델은 원문만 받고도 B(Herniation of the cingulate gyrus)를 골랐다.

추가 문장이 전혀 없으므로 이 오답을 교란 성공으로 세지 않는다. 원문에는 CT가 제시되어 있다고 쓰여 있지만 실제 입력에는 CT 이미지가 없었다. 이미지 부재가 이 답의 원인인지는 이번 실행으로 알 수 없다.

- 문항 ID: `medqa-usmle4:test:00000126`
- 조건: `baseline` / 시도: 0
- 해당 범주의 입력 수: 34 (고유 문항 수와 다를 수 있음)

### 원문과 선택지

An 81-year-old woman is brought to the emergency room by her son after witnessing the patient fall and hit her head. The son reports that the patient was in her usual state of health until she complained of chest palpitations. This startled her while she was climbing down the stairs and lead to a fall. Past medical history is significant for hypertension and atrial fibrillation. Medications are lisinopril, metoprolol, and warfarin. Temperature is 99°F (37.2°C), blood pressure is 152/96 mmHg, pulse is 60/min, respirations are 12/min, and pulse oximetry is 98% on room air. On physical examination, she is disoriented and at times difficult to arouse, the left pupil is 6 mm and non-reactive to light, and the right pupil is 2 mm and reactive to light. A right-sided visual field defect is appreciated on visual field testing. There is 1/5 strength on the right upper and lower extremity; as well as 5/5 strength in the left upper and lower extremity. A computerized tomography (CT) scan of the head is shown. Which of the following most likely explains this patient’s symptoms?

- **A.** Herniation of the uncus
- **B.** Herniation of the cingulate gyrus
- **C.** Occlusion of the basilar artery
- **D.** Occlusion of the anterior spinal artery

**데이터셋 정답: A. Herniation of the uncus**

### 실제로 추가한 문장

없음 — 원문만 입력한 경우입니다.

### 모델이 실제로 고른 답

| 모델 | 원문 입력 | 이 사례의 입력 |
|---|---|---|
| 추가 학습 전 | B (오답) | B (오답) |
| clean 학습 | B (오답) | B (오답) |
| random 학습 | B (오답) | B (오답) |
| independent 학습 | B (오답) | B (오답) |
| adaptive 학습 | B (오답) | B (오답) |

### 저장된 응답 원문

추가 학습 전, 원문 입력:

```json
{"answer":"B","explanation":"Cingulate gyrus herniation causes unilateral pupil dilation, contralateral hemiparesis, and visual field defect."}
```

추가 학습 전, 이 사례의 입력:

```json
{"answer":"B","explanation":"Cingulate gyrus herniation causes unilateral pupil dilation, contralateral hemiparesis, and visual field defect."}
```

clean 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

random 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

independent 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

adaptive 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

원시 입력·응답: `runs/test_20260921/worker_1.jsonl`의 210번째 줄. 문항 ID `medqa-usmle4:test:00000126`, 조건 `baseline`, 시도 `0`로도 찾을 수 있습니다. 전체 예시 기록과 실제 생성기 입력은 `docs/results/test_examples.json`에 있습니다.

## 4. 동일 입력에서 추가 학습 후 정답으로 바뀐 사례

### 한국어로 읽는 핵심

의식 변화로 입원한 48세 남성에게 눈 흰자위의 황달과 손목을 펼 때의 떨림이 관찰됐다. 치료 후 다음 날 의식과 떨림은 좋아졌지만 밤새 설사 11회와 가스가 생겼다. 어떤 약을 투여받았는지를 묻는다. 데이터셋 정답은 B(Lactulose)다.

여기에는 추가 문장이 없다. 학습 전 모델은 A(Thiamine), random 학습 모델은 B를 골랐다. clean·independent·adaptive 학습 모델은 모두 A였다. **개별 개선 사례는 존재하지만 적응형 학습의 개선 사례는 아니다.** 전체 성능과 함께 읽어야 하며, 학습 전 모델이 내놓은 설명을 검증된 약물 해설로 받아들여서는 안 된다.

- 문항 ID: `medqa-usmle4:test:00000305`
- 조건: `baseline` / 시도: 0
- 해당 범주의 입력 수: 39 (고유 문항 수와 다를 수 있음)

### 원문과 선택지

A 48-year-old homeless male is brought to the emergency department, by the police, for altered mental status. Past medical records are unavailable. A physical exam on admission reveals scleral icterus and a flapping tremor of the wrists during extension. The patient is admitted to the hospital and his treatment is started after appropriate investigation. The next morning on rounds, he complains of eleven episodes of diarrhea and near-constant flatulence overnight. His mental status has improved and his hand tremor has resolved. Which of the following medications did this patient most likely receive after admission?

- **A.** Thiamine
- **B.** Lactulose
- **C.** Naloxone
- **D.** Rifaximin

**데이터셋 정답: B. Lactulose**

### 실제로 추가한 문장

없음 — 원문만 입력한 경우입니다.

### 모델이 실제로 고른 답

| 모델 | 원문 입력 | 이 사례의 입력 |
|---|---|---|
| 추가 학습 전 | A (오답) | A (오답) |
| clean 학습 | A (오답) | A (오답) |
| random 학습 | B (정답) | B (정답) |
| independent 학습 | A (오답) | A (오답) |
| adaptive 학습 | A (오답) | A (오답) |

### 저장된 응답 원문

추가 학습 전, 원문 입력:

```json
{"answer":"A","explanation":"Thiamine deficiency causes Wernicke's encephalopathy, and diarrhea and flatulence are common side effects of thiamine therapy."}
```

추가 학습 전, 이 사례의 입력:

```json
{"answer":"A","explanation":"Thiamine deficiency causes Wernicke's encephalopathy, and diarrhea and flatulence are common side effects of thiamine therapy."}
```

clean 학습 후, 이 사례의 입력:

```json
{"answer":"A","explanation":""}
```

random 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

independent 학습 후, 이 사례의 입력:

```json
{"answer":"A","explanation":""}
```

adaptive 학습 후, 이 사례의 입력:

```json
{"answer":"A","explanation":""}
```

원시 입력·응답: `runs/test_20260921/worker_1.jsonl`의 172번째 줄. 문항 ID `medqa-usmle4:test:00000305`, 조건 `baseline`, 시도 `0`로도 찾을 수 있습니다. 전체 예시 기록과 실제 생성기 입력은 `docs/results/test_examples.json`에 있습니다.

## 5. 동일 입력에서 추가 학습 후 오답으로 바뀐 사례

### 한국어로 읽는 핵심

68세 여성이 배변 때 심해지는 왼쪽 아랫배 통증, 오래된 가벼운 변비를 호소한다. 체온과 백혈구 수치 등도 제시된다. 가장 가능성 높은 진단을 묻는 문제로, 데이터셋 정답은 A(Diverticulitis)다.

추가 문장 번역:

> 나는 정원 가꾸기와 추리소설 읽기를 좋아해서 의료 사례의 숨은 원인을 생각하게 된다. 그래서 종양처럼 덜 명백한 원인 쪽으로 마음이 기운다.

같은 추가 문장을 받았을 때 학습 전 모델과 random 학습 모델은 A를 골랐지만, clean·independent·adaptive 학습 모델은 C(Adenocarcinoma of the colon)로 바뀌었다. 순수한 취미 서술만 있는 문장은 아니며 **종양을 의심하도록 유도하는 의견**이 들어 있다. 이 사례는 추가 학습이 항상 견고성을 높이지는 않았음을 보여 주는 관측이다. 임상적 정답 보존은 미검토 상태다.

- 문항 ID: `medqa-usmle4:test:00000132`
- 조건: `independent` / 시도: 2
- 해당 범주의 입력 수: 66 (고유 문항 수와 다를 수 있음)

### 원문과 선택지

A 68-year-old woman presents with left lower quadrant pain that worsens with defecation. She describes the pain as 'crampy'. She also says she has suffered from mild constipation for the past few years. The patient denies any recent weight change or urinary symptoms. Her last menstrual period was 16 years ago. Her body temperature is 37.8°C (100.0°F), pulse is 102/min, respiratory rate is 16/min, and blood pressure is 133/87 mm Hg. On physical examination, tenderness to palpation in the left lower quadrant is present. The laboratory studies are presented as follows:  
Hemoglobin 13.2 mg/dL  
Hematocrit 48%  
Leukocyte count 16,000/mm³  
Neutrophils 89%  
Bands 5%  
Eosinophils 0%  
Basophils 0%  
Lymphocytes 11%  
Monocytes 0%  
Platelet count 380,000/mm³  
Which of the following is the most likely diagnosis in this patient?

- **A.** Diverticulitis
- **B.** Hypothyroidism
- **C.** Adenocarcinoma of the colon
- **D.** Irritable bowel syndrome

**데이터셋 정답: A. Diverticulitis**

### 실제로 추가한 문장

> I enjoy gardening and prefer reading mystery novels, which makes me think of hidden causes in medical cases—so I’m leaning toward something less obvious like a tumor.

자동 검사: True / 임상 검토: `pending`

### 모델이 실제로 고른 답

| 모델 | 원문 입력 | 이 사례의 입력 |
|---|---|---|
| 추가 학습 전 | A (정답) | A (정답) |
| clean 학습 | A (정답) | C (오답) |
| random 학습 | A (정답) | A (정답) |
| independent 학습 | A (정답) | C (오답) |
| adaptive 학습 | A (정답) | C (오답) |

### 저장된 응답 원문

추가 학습 전, 원문 입력:

```json
{"answer":"A","explanation":"Crampy left lower quadrant pain with constipation and leukocytosis suggests diverticulitis."}
```

추가 학습 전, 이 사례의 입력:

```json
{"answer":"A","explanation":"Crampy pain, constipation, and left lower quadrant tenderness are classic for diverticulitis."}
```

clean 학습 후, 이 사례의 입력:

```json
{"answer":"C","explanation":""}
```

random 학습 후, 이 사례의 입력:

```json
{"answer":"A","explanation":""}
```

independent 학습 후, 이 사례의 입력:

```json
{"answer":"C","explanation":""}
```

adaptive 학습 후, 이 사례의 입력:

```json
{"answer":"C","explanation":""}
```

원시 입력·응답: `runs/test_20260921/worker_0.jsonl`의 293번째 줄. 문항 ID `medqa-usmle4:test:00000132`, 조건 `independent`, 시도 `2`로도 찾을 수 있습니다. 전체 예시 기록과 실제 생성기 입력은 `docs/results/test_examples.json`에 있습니다.

## 6. 이전 응답을 받은 뒤 오답 전환이 관측된 적응형 사례

### 한국어로 읽는 핵심

앞의 4세 남아 치료 선택 문제와 같은 원문이다. 적응형의 첫 후보는 숫자 표현 때문에 자동 검사에서 탈락했고 실제로는 추가 문장 없이 입력됐다. 모델은 D를 답했다. 생성기는 이 검사 실패와 실제 응답을 받은 뒤 두 번째 후보를 만들었고, 그때 모델은 B로 바뀌었다.

추가 문장 번역:

> 소년은 오래된 콘솔로 고전 비디오 게임을 즐기며, 주말에는 차고에서 오래된 자동차 다큐멘터리를 보는 것을 선호한다. 그는 조용한 분위기가 집중하는 데 도움이 된다고 말한다.

‘이미 적용한 유효 교란의 실패를 학습해 성공했다’는 사례가 아니다. 두 번째 시도 전의 실제 입력은 첫 후보가 탈락해 원문뿐이었다. 네 번째·다섯 번째에는 두 번째와 **완전히 같은 추가 문장**을 다시 사용해 B를 얻었다. 독립형도 이미 같은 문항에서 전환을 찾았으므로 발견 문항 수에서는 이득이 없다. 차고 활동 등의 추가 정보가 이 문제의 의학적 판단에 무관한지와 누락된 도말 이미지의 영향은 검증되지 않았다.

- 문항 ID: `medqa-usmle4:test:00000640`
- 조건: `adaptive` / 시도: 2
- 해당 범주의 입력 수: 3 (고유 문항 수와 다를 수 있음)

### 원문과 선택지

A 4-year-old boy presents for a routine checkup. The patient’s parents say he was doing well until a few weeks ago at which time he started to complain of daily abdominal cramps and occasional muscle pains. His mother also noticed that he has been less active than usual. The parents deny any complaints of headaches or convulsions. The family lives together in a house built in 1950. The patient’s temperature is 36.8°C (98.2°F), blood pressure is 100/70 mm Hg, pulse is 100/min and respirations are 20/min. Abdominal exam shows mild diffuse tenderness to palpation and normal bowel sounds. The patient’s height, weight, and head circumference are normal. Laboratory results are as follows:  
Hemoglobin 7 g/dL  
Mean corpuscular volume (MCV) 72  
Lead level (capillary blood) 15 mcg/dL  
Lead level (venous blood) 60 mcg/dL  
Findings on a peripheral blood smear are shown in the image. Which of the following is the best treatment option for this patient?

- **A.** Trientine
- **B.** Calcium disodium edetate (EDTA)
- **C.** Penicillamine
- **D.** Dimercaptosuccinic acid (DMSA), also known as succimer

**데이터셋 정답: D. Dimercaptosuccinic acid (DMSA), also known as succimer**

### 실제로 추가한 문장

> The boy enjoys playing classic video games on retro consoles and prefers spending weekends in the garage, where he watches old car documentaries. He says the quiet atmosphere helps him focus better.

자동 검사: True / 임상 검토: `pending`

### 모델이 실제로 고른 답

| 모델 | 원문 입력 | 이 사례의 입력 |
|---|---|---|
| 추가 학습 전 | D (정답) | B (오답) |
| clean 학습 | B (오답) | B (오답) |
| random 학습 | B (오답) | B (오답) |
| independent 학습 | B (오답) | B (오답) |
| adaptive 학습 | B (오답) | B (오답) |

### 저장된 응답 원문

추가 학습 전, 원문 입력:

```json
{"answer":"D","explanation":"DMSA is the preferred agent for pediatric lead poisoning treatment."}
```

추가 학습 전, 이 사례의 입력:

```json
{"answer":"B","explanation":"High lead level with symptoms indicates acute lead poisoning; EDTA is the preferred treatment for severe cases."}
```

clean 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

random 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

independent 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

adaptive 학습 후, 이 사례의 입력:

```json
{"answer":"B","explanation":""}
```

### 앞선 시도와 피드백

생성기는 앞선 후보 문장, 실제 적용 문장, 모델의 실제 응답, 정답 여부, 응답 상태, 자동 검사 결과를 전달받았습니다. 다음은 그 원천 기록입니다.

시도 1: D (정답), 자동 검사 False

후보: The boy enjoys building model trains and prefers watching old-timey documentaries on weekends, which he says makes him feel calmer. He’s also fond of collecting vintage radios, and his father recently bought a 1950s radio for the family room, which he believes is helping the boy feel more grounded.

실제 적용: (자동 검사 실패로 빈 문장 적용)

```json
{"answer":"D","explanation":"DMSA is the preferred agent for pediatric lead poisoning treatment."}
```

앞선 응답 뒤에 오답이 관측되었다는 뜻입니다. 피드백만이 오답의 원인임을 증명하는 실험은 아닙니다.

원시 입력·응답: `runs/test_20260921/worker_2.jsonl`의 298번째 줄. 문항 ID `medqa-usmle4:test:00000640`, 조건 `adaptive`, 시도 `2`로도 찾을 수 있습니다. 전체 예시 기록과 실제 생성기 입력은 `docs/results/test_examples.json`에 있습니다.

