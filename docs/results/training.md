# 네 조건 QLoRA 학습 기록

상태: complete. 공통 train 문항 10개, 조건당 20행, optimizer 20회.

해석 범위: unreviewed_exploratory. 미검토 선택 문장 30개.

clean은 원문을 두 번, 나머지 조건은 원문과 선택한 추가 문장을 한 번씩 사용했습니다. 모든 조건은 같은 source gold 답만 감독했습니다. loss 감소는 견고성 개선의 증거가 아닙니다. 별도 held-out 평가로 효과를 확인해야 합니다.
