# 파일럿 점검 결과

이 실행은 파일럿입니다. 파이프라인과 오류를 점검하는 자료이며 최종 성능 비교로 해석하지 않습니다.

문항 20개 중 baseline 정답 집합 S는 9개입니다. clean_repeat·independent·adaptive의 예정 target 호출 예산은 각각 K=5입니다.

literal_control은 고정 문장 3개를 점검하는 별도 보조 조건입니다. 주조건의 동등 예산 비교와 분리하여 읽어야 합니다.

전체 문항 clean 정확도: 45.0% (9/20).

| 조건 | 호출 예산 | 전체 슬롯 정확도 | S 내 슬롯 정확도 | 예산 내 구문 기준 전환 | 예산 내 승인 사례 | 거절 / 형식 / 실행 오류 / 누락 |
|---|---:|---:|---:|---:|---:|---:|
| clean_repeat | 5 | 45.0% | 100.0% | 0.0% | 0.0% | 0 / 0 / 0 / 0 |
| independent | 5 | 45.0% | 100.0% | 0.0% | 0.0% | 0 / 0 / 0 / 0 |
| adaptive | 5 | 42.0% | 93.3% | 22.2% | 0.0% | 0 / 0 / 0 / 0 |
| literal_control | 3 | 45.0% | 100.0% | 0.0% | 0.0% | 0 / 0 / 0 / 0 |

정확도의 분모는 예정 호출 슬롯 전체입니다. 응답 거절과 형식 오류는 별도 표시하며 오답 선택지로 취급하지 않습니다. @K는 같은 S의 각 문항에서 K회 이내 오답 선택지가 한 번 이상 나온 비율입니다. 시도 횟수를 독립 문항 수로 세지 않습니다.

구문 기준 값은 임상적 정답 보존이 확인된 공격 성공률이 아닙니다. 승인 사례 값은 review.csv에서 approved인 변형만 셉니다. 미검토 사례가 남아 있는 동안 이 값은 확정 사례의 하한입니다. clean_repeat에는 새로운 임상 변형이 없으며 주로 원문 재호출 변동을 확인합니다.

임상 검토: 승인 0, 기각 0, 대기 260; 검토율 0.0%. review.csv의 clinical_review를 pending/approved/rejected로 수정하고 다시 집계하면 판정과 메모가 유지됩니다.

구문 기준 adaptive − independent @K: +22.2%p (문항 단위 paired bootstrap 95% CI +0.0~+55.6%p, 문항 9개, seed 20260921).
검토 승인 기준 adaptive − independent @K: +0.0%p (문항 단위 paired bootstrap 95% CI +0.0~+0.0%p, 문항 9개, seed 20260921).

K별 경과

| 조건 | k | 구문 기준 전환 문항 / S | 승인된 전환 문항 / S | 누락 슬롯 |
|---|---:|---:|---:|---:|
| clean_repeat | 1 | 0 / 9 | 0 / 9 | 0 |
| clean_repeat | 2 | 0 / 9 | 0 / 9 | 0 |
| clean_repeat | 3 | 0 / 9 | 0 / 9 | 0 |
| clean_repeat | 4 | 0 / 9 | 0 / 9 | 0 |
| clean_repeat | 5 | 0 / 9 | 0 / 9 | 0 |
| independent | 1 | 0 / 9 | 0 / 9 | 0 |
| independent | 2 | 0 / 9 | 0 / 9 | 0 |
| independent | 3 | 0 / 9 | 0 / 9 | 0 |
| independent | 4 | 0 / 9 | 0 / 9 | 0 |
| independent | 5 | 0 / 9 | 0 / 9 | 0 |
| adaptive | 1 | 1 / 9 | 0 / 9 | 0 |
| adaptive | 2 | 2 / 9 | 0 / 9 | 0 |
| adaptive | 3 | 2 / 9 | 0 / 9 | 0 |
| adaptive | 4 | 2 / 9 | 0 / 9 | 0 |
| adaptive | 5 | 2 / 9 | 0 / 9 | 0 |
| literal_control | 1 | 0 / 9 | 0 / 9 | 0 |
| literal_control | 2 | 0 / 9 | 0 / 9 | 0 |
| literal_control | 3 | 0 / 9 | 0 / 9 | 0 |
