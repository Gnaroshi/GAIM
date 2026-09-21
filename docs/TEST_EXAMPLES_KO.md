# 실제 시험 예시 6개

**기존 4비트 추가 학습 때 얻은 실제 결과**입니다. 새 BF16 학습 결과가 아닙니다. 정답은 MedQA 정답표를 따르며, 모델이 쓴 해설을 검증된 의학 설명으로 사용하지 않습니다.

모든 모델의 출발점은 **Qwen 팀이 이미 학습해 공개한 Qwen3-4B-Instruct-2507**입니다. “추가 학습 없음”은 그 공개 모델 그대로입니다. 다른 네 모델에는 같은 의료 문제 10개를 아래 방식으로 추가 학습했습니다.

| 아래 표의 이름 | 추가 학습 입력 20개 |
|---|---|
| 원문 학습 | 원문 10개를 두 번 |
| 고정 취미 문장 학습 | 원문 10개 + 취미 문장을 붙인 10개 |
| 이전 답 없이 만든 문장 학습 | 원문 10개 + 이전 답을 보지 않고 생성한 문장을 붙인 10개 |
| 이전 답을 보고 만든 문장 학습 | 원문 10개 + 이전 답과 정오를 보고 생성한 문장을 붙인 10개 |

네 조건 모두 원본 정답을 가르쳤습니다. 문장 생성 방법은 다르지만 별도의 사전학습 모델을 사용한 것은 아닙니다. 각 범주에서 문항 ID·시도 순서로 첫 사례를 골랐으며, 전체 성능은 [시험 결과](TEST_RESULTS_KO.md)를 보세요. 모든 추가 문장은 의학적 검토 대기입니다.

## 1. 고정 배경 문장을 넣어도 정답을 유지한 사례

22번 염색체의 merlin 관련 유전자 변이와 질환 위험을 묻는 문제입니다. 정답은 B입니다. Porsche 취미 문장을 넣어도 공개 모델과 네 학습 모델 모두 B를 유지했습니다. LoL·mango 고정 문장에서도 같았습니다.

문항 `medqa-usmle4:test:00000012` · 고정 취미 문장 · 시도 1 · **정답 B**

<details>
<summary>의료 문제 원문과 선택지</summary>

A 20-year-old man comes to the physician because of worsening gait unsteadiness and bilateral hearing loss for 1 month. He has had intermittent tingling sensations on both cheeks over this time period. He has no history of serious medical illness and takes no medications. Audiometry shows bilateral sensorineural hearing loss. Genetic evaluation shows a mutation of a tumor suppressor gene on chromosome 22 that encodes merlin. This patient is at increased risk for which of the following conditions?

- A. Renal cell carcinoma
- B. Meningioma
- C. Astrocytoma
- D. Vascular malformations

</details>

**실제로 추가한 문장**

> The patient enjoys browsing Porsche car brochures.

| 모델 | 이 입력에 대한 답 |
|---|---|
| 추가 학습 없음 | B |
| 원문 학습 | B |
| 고정 취미 문장 학습 | B |
| 이전 답 없이 만든 문장 학습 | B |
| 이전 답을 보고 만든 문장 학습 | B |

**공개 모델의 실제 응답**

```json
{"answer":"B","explanation":"Mutation in merlin gene (NF2) is associated with meningioma risk."}
```

## 2. 원문 정답에서 문장을 넣은 뒤 오답으로 바뀐 잠정 사례

4세 남아의 검사 결과를 보고 치료를 고르는 문제입니다. 공개 모델은 원문에 D, 추가 문장이 있을 때 B를 답했습니다. 그러나 추가 문장에는 **약물명 DMSA와 의학적 설명**이 들어갑니다. 원문이 가리키는 혈액 사진도 입력에 없습니다. 의학적으로 무관한 취미 때문에 틀렸다고 확정할 수 없습니다.

문항 `medqa-usmle4:test:00000640` · 이전 답 없이 생성 · 시도 1 · **정답 D**

<details>
<summary>의료 문제 원문과 선택지</summary>

A 4-year-old boy presents for a routine checkup. The patient’s parents say he was doing well until a few weeks ago at which time he started to complain of daily abdominal cramps and occasional muscle pains. His mother also noticed that he has been less active than usual. The parents deny any complaints of headaches or convulsions. The family lives together in a house built in 1950. The patient’s temperature is 36.8°C (98.2°F), blood pressure is 100/70 mm Hg, pulse is 100/min and respirations are 20/min. Abdominal exam shows mild diffuse tenderness to palpation and normal bowel sounds. The patient’s height, weight, and head circumference are normal. Laboratory results are as follows:
Hemoglobin 7 g/dL
Mean corpuscular volume (MCV) 72
Lead level (capillary blood) 15 mcg/dL
Lead level (venous blood) 60 mcg/dL
Findings on a peripheral blood smear are shown in the image. Which of the following is the best treatment option for this patient?

- A. Trientine
- B. Calcium disodium edetate (EDTA)
- C. Penicillamine
- D. Dimercaptosuccinic acid (DMSA), also known as succimer

</details>

**실제로 추가한 문장**

> The boy enjoys building model trains and often spends hours indoors playing with them, which he says makes him less active. He prefers DMSA over other options because it's more convenient for home use, though this is just a personal preference.

| 모델 | 이 입력에 대한 답 |
|---|---|
| 추가 학습 없음 | B |
| 원문 학습 | B |
| 고정 취미 문장 학습 | B |
| 이전 답 없이 만든 문장 학습 | B |
| 이전 답을 보고 만든 문장 학습 | B |

**공개 모델의 실제 응답**

```json
{"answer":"B","explanation":"High lead level indicates acute lead poisoning; EDTA is the preferred treatment for severe cases."}
```

원문만 받은 공개 모델의 실제 응답:

```json
{"answer":"D","explanation":"DMSA is the preferred agent for pediatric lead poisoning treatment."}
```

## 3. 추가 문장 없이도 처음부터 틀린 사례

81세 여성의 낙상 후 증상 원인을 묻습니다. 정답은 A지만 공개 모델은 추가 문장 없이도 B를 골랐습니다. 따라서 교란 성공 사례가 아닙니다. 원문에 언급된 CT 이미지도 입력에 없습니다.

문항 `medqa-usmle4:test:00000126` · 추가 문장 없음 · 시도 0 · **정답 A**

<details>
<summary>의료 문제 원문과 선택지</summary>

An 81-year-old woman is brought to the emergency room by her son after witnessing the patient fall and hit her head. The son reports that the patient was in her usual state of health until she complained of chest palpitations. This startled her while she was climbing down the stairs and lead to a fall. Past medical history is significant for hypertension and atrial fibrillation. Medications are lisinopril, metoprolol, and warfarin. Temperature is 99°F (37.2°C), blood pressure is 152/96 mmHg, pulse is 60/min, respirations are 12/min, and pulse oximetry is 98% on room air. On physical examination, she is disoriented and at times difficult to arouse, the left pupil is 6 mm and non-reactive to light, and the right pupil is 2 mm and reactive to light. A right-sided visual field defect is appreciated on visual field testing. There is 1/5 strength on the right upper and lower extremity; as well as 5/5 strength in the left upper and lower extremity. A computerized tomography (CT) scan of the head is shown. Which of the following most likely explains this patient’s symptoms?

- A. Herniation of the uncus
- B. Herniation of the cingulate gyrus
- C. Occlusion of the basilar artery
- D. Occlusion of the anterior spinal artery

</details>

**실제로 추가한 문장**

없음. 원문만 입력했습니다.

| 모델 | 이 입력에 대한 답 |
|---|---|
| 추가 학습 없음 | B |
| 원문 학습 | B |
| 고정 취미 문장 학습 | B |
| 이전 답 없이 만든 문장 학습 | B |
| 이전 답을 보고 만든 문장 학습 | B |

**공개 모델의 실제 응답**

```json
{"answer":"B","explanation":"Cingulate gyrus herniation causes unilateral pupil dilation, contralateral hemiparesis, and visual field defect."}
```

## 4. 동일 입력에서 추가 학습 후 정답으로 바뀐 사례

48세 남성이 치료 뒤 의식은 좋아졌지만 설사를 했다는 문제입니다. 투여한 약을 묻고 정답은 B(Lactulose)입니다. 공개 모델은 A, 고정 취미 문장으로 학습한 모델은 B를 골랐습니다. 이 한 사례의 개선이 전체 성능 개선을 뜻하지는 않습니다.

문항 `medqa-usmle4:test:00000305` · 추가 문장 없음 · 시도 0 · **정답 B**

<details>
<summary>의료 문제 원문과 선택지</summary>

A 48-year-old homeless male is brought to the emergency department, by the police, for altered mental status. Past medical records are unavailable. A physical exam on admission reveals scleral icterus and a flapping tremor of the wrists during extension. The patient is admitted to the hospital and his treatment is started after appropriate investigation. The next morning on rounds, he complains of eleven episodes of diarrhea and near-constant flatulence overnight. His mental status has improved and his hand tremor has resolved. Which of the following medications did this patient most likely receive after admission?

- A. Thiamine
- B. Lactulose
- C. Naloxone
- D. Rifaximin

</details>

**실제로 추가한 문장**

없음. 원문만 입력했습니다.

| 모델 | 이 입력에 대한 답 |
|---|---|
| 추가 학습 없음 | A |
| 원문 학습 | A |
| 고정 취미 문장 학습 | B |
| 이전 답 없이 만든 문장 학습 | A |
| 이전 답을 보고 만든 문장 학습 | A |

**공개 모델의 실제 응답**

```json
{"answer":"A","explanation":"Thiamine deficiency causes Wernicke's encephalopathy, and diarrhea and flatulence are common side effects of thiamine therapy."}
```

## 5. 동일 입력에서 추가 학습 후 오답으로 바뀐 사례

68세 여성의 왼쪽 아랫배 통증과 검사 결과로 진단을 고르는 문제입니다. 정답은 A입니다. 추가 문장에는 종양을 의심하도록 유도하는 의견이 있습니다. 같은 입력에서 공개 모델은 A, 일부 추가 학습 모델은 C로 답했습니다. 순수한 취미 정보만 넣은 사례는 아닙니다.

문항 `medqa-usmle4:test:00000132` · 이전 답 없이 생성 · 시도 2 · **정답 A**

<details>
<summary>의료 문제 원문과 선택지</summary>

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

- A. Diverticulitis
- B. Hypothyroidism
- C. Adenocarcinoma of the colon
- D. Irritable bowel syndrome

</details>

**실제로 추가한 문장**

> I enjoy gardening and prefer reading mystery novels, which makes me think of hidden causes in medical cases—so I’m leaning toward something less obvious like a tumor.

| 모델 | 이 입력에 대한 답 |
|---|---|
| 추가 학습 없음 | A |
| 원문 학습 | C |
| 고정 취미 문장 학습 | A |
| 이전 답 없이 만든 문장 학습 | C |
| 이전 답을 보고 만든 문장 학습 | C |

**공개 모델의 실제 응답**

```json
{"answer":"A","explanation":"Crampy pain, constipation, and left lower quadrant tenderness are classic for diverticulitis."}
```

## 6. 이전 응답을 받은 뒤 오답 전환이 관측된 사례

2번과 같은 문제입니다. 첫 생성 문장은 검사에서 탈락해 실제로는 원문만 입력했고 D를 받았습니다. 생성기가 이 결과를 본 뒤 두 번째 문장을 만들자 B가 나왔습니다. 앞선 “유효한 공격의 실패”를 고친 사례는 아닙니다. 차고 활동이라는 추가 정보의 의학적 무관성도 확인되지 않았습니다.

문항 `medqa-usmle4:test:00000640` · 이전 답을 보고 생성 · 시도 2 · **정답 D**

**실제로 추가한 문장**

> The boy enjoys playing classic video games on retro consoles and prefers spending weekends in the garage, where he watches old car documentaries. He says the quiet atmosphere helps him focus better.

| 모델 | 이 입력에 대한 답 |
|---|---|
| 추가 학습 없음 | B |
| 원문 학습 | B |
| 고정 취미 문장 학습 | B |
| 이전 답 없이 만든 문장 학습 | B |
| 이전 답을 보고 만든 문장 학습 | B |

**공개 모델의 실제 응답**

```json
{"answer":"B","explanation":"High lead level with symptoms indicates acute lead poisoning; EDTA is the preferred treatment for severe cases."}
```

원문만 받은 공개 모델의 실제 응답:

```json
{"answer":"D","explanation":"DMSA is the preferred agent for pediatric lead poisoning treatment."}
```

두 번째 시도의 생성기가 받은 피드백은 첫 문장의 검사 탈락과 실제 답 D였습니다. 네 번째·다섯 번째에도 두 번째와 같은 문장으로 B를 얻었습니다. 이전 답을 보지 않는 방식도 같은 문제에서 이미 전환을 찾았으므로 발견한 문제 수가 늘지는 않았습니다.

## 전체 입력과 응답

[예시별 원문 JSON](results/test_examples.json)에 모든 모델의 응답과 생성기에 전달한 이력이 있습니다. 전체 시험 기록은 [repro/reference](../repro/reference/README.md)에 있습니다.
